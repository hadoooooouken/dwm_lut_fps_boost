// dllmain.cpp : Defines the entry point for the DLL application.

#ifndef _DLL
#endif

#include "pch.h"

#include <io.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <DirectXMath.h>
#include <immintrin.h>
#include <intrin.h>
#pragma comment (lib, "d3d11.lib")
#pragma comment (lib, "d3dcompiler.lib")
#pragma comment (lib, "dxgi.lib")
#pragma comment (lib, "uuid.lib")
#pragma comment (lib, "dxguid.lib")

#pragma intrinsic(_ReturnAddress)

#define DITHER_GAMMA 2.2f
#define LUT_FOLDER "%SYSTEMROOT%\\Temp\\luts"

#define RELEASE_IF_NOT_NULL(x) { if ((x) != NULL) { (x)->Release(); (x) = NULL; } }
#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)
#define RESIZE(x, y) \
    do { \
        void* _temp = realloc((x), (y) * sizeof(*(x))); \
        if (_temp != NULL) { \
            (x) = (decltype(x))_temp; \
        } else { \
            if ((y) > 0) { \
                free((x)); \
                (x) = NULL; \
            } \
            if (DEBUG_MODE) { \
                log_to_file("Memory allocation failed in RESIZE macro"); \
            } \
        } \
    } while(0)
#define LOG_FILE_PATH R"(C:\DWMLOG\dwm.log)"
#define MAX_LOG_FILE_SIZE 20 * 1024 * 1024
#ifdef _DEBUG
#define DEBUG_MODE true
#else
#define DEBUG_MODE false
#endif

#if DEBUG_MODE == true
#define __LOG_ONLY_ONCE(x, y) if (static bool first_log_##y = true) { log_to_file(x); first_log_##y = false; }
#define _LOG_ONLY_ONCE(x, y) __LOG_ONLY_ONCE(x, y)
#define LOG_ONLY_ONCE(x) _LOG_ONLY_ONCE(x, __COUNTER__)
#define MESSAGE_BOX_DBG(x, y) MessageBoxA(NULL, x, "DEBUG HOOK DWM", y);

#define EXECUTE_WITH_LOG(winapi_func_hr) \
	do { \
		HRESULT hr = (winapi_func_hr); \
		if (FAILED(hr)) \
		{ \
			std::stringstream ss; \
			ss << "ERROR AT LINE: " << __LINE__ << " HR: " << hr << " - DETAILS: "; \
			LPSTR error_message = nullptr; \
			FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, \
				NULL, hr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&error_message, 0, NULL); \
			ss << error_message; \
			log_to_file(ss.str().c_str()); \
			LocalFree(error_message); \
			throw std::exception(ss.str().c_str()); \
		} \
	} while (false);

#define EXECUTE_D3DCOMPILE_WITH_LOG(winapi_func_hr, error_interface) \
	do { \
		HRESULT hr = (winapi_func_hr); \
		if (FAILED(hr)) \
		{ \
			std::stringstream ss; \
			ss << "ERROR AT LINE: " << __LINE__ << " HR: " << hr << " - DETAILS: "; \
			LPSTR error_message = nullptr; \
			FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, \
				NULL, hr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&error_message, 0, NULL); \
			ss << error_message << " - DX COMPILE ERROR: " << (char*)error_interface->GetBufferPointer(); \
			error_interface->Release(); \
			log_to_file(ss.str().c_str()); \
			LocalFree(error_message); \
			throw std::exception(ss.str().c_str()); \
		} \
	} while (false);

#define LOG_ADDRESS(prefix_message, address) \
	{ \
		std::stringstream ss; \
		ss << prefix_message << " 0x" << std::setw(sizeof(address) * 2) << std::setfill('0') << std::hex << (UINT_PTR)address; \
		log_to_file(ss.str().c_str()); \
	}

#else
#define LOG_ONLY_ONCE(x) // NOP, not in debug mode
#define MESSAGE_BOX_DBG(x, y) // NOP, not in debug mode
#define EXECUTE_WITH_LOG(winapi_func_hr) winapi_func_hr;
#define EXECUTE_D3DCOMPILE_WITH_LOG(winapi_func_hr, error_interface) winapi_func_hr;
#define LOG_ADDRESS(prefix_message, address) // NOP, not in debug mode
#endif

#if DEBUG_MODE == true
void print_error(const char* prefix_message)
{
	DWORD errorCode = GetLastError();
	LPSTR errorMessage = nullptr;
	FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		nullptr, errorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&errorMessage, 0, nullptr);

	char message_buf[100];
	sprintf_s(message_buf, sizeof(message_buf), "%s: %s - error code: %u", prefix_message, errorMessage, errorCode);
	MESSAGE_BOX_DBG(message_buf, MB_OK | MB_ICONWARNING)
	LocalFree(errorMessage);
}

void log_to_file(const char* log_buf)
{
	FILE* pFile = nullptr;
	fopen_s(&pFile, LOG_FILE_PATH, "a");
	if (pFile == NULL)
	{
		return;
	}
	fseek(pFile, 0, SEEK_END);
	long size = ftell(pFile);
	if (size > MAX_LOG_FILE_SIZE)
	{
		if (_chsize_s(_fileno(pFile), 0) != 0)
		{
			fclose(pFile);
			return;
		}
	}
	fseek(pFile, 0, SEEK_END);
	fprintf(pFile, "%s\n", log_buf);
	fclose(pFile);
}
#endif

// AVX2 optimized float to half conversion batch
__forceinline void FloatToHalfBatch_AVX2(const float* src, uint16_t* dst, size_t count) {
    size_t i = 0;

    // Process 8 floats at a time with AVX2
    for (; i + 7 < count; i += 8) {
        // Always use unaligned loads for safety
        __m256 vf = _mm256_loadu_ps(src + i);
        __m128i vh = _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i*)(dst + i), vh);
    }

    // Process 4 floats at a time
    for (; i + 3 < count; i += 4) {
        __m128 vf = _mm_loadu_ps(src + i);
        __m128i vh = _mm_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i*)(dst + i), vh);
    }

    // Process remaining 0-3 elements
    for (; i < count; i++) {
        __m128 vf = _mm_set_ss(src[i]);
        __m128i vh = _mm_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
        dst[i] = (uint16_t)_mm_cvtsi128_si32(vh);
    }
}

// AVX2 optimized noise texture conversion with aligned loads
__forceinline void ConvertNoiseBytesToFloat_AVX2(const unsigned char noiseBytes[NOISE_SIZE][NOISE_SIZE], 
                                               float* output) {
    // Check that the array is 32-byte aligned
    assert((uintptr_t)&noiseBytes[0][0] % 32 == 0 && "Noise texture should be 32-byte aligned");
    
    const uint8_t* src = &noiseBytes[0][0];
    const size_t total = NOISE_SIZE * NOISE_SIZE;
    
    const __m256 add_const = _mm256_set1_ps(0.5f);
    const __m256 div_const = _mm256_set1_ps(1.0f / 256.0f);
    
    size_t i = 0;
    
    // Process 32 bytes at a time (8 sets of 4 bytes) with AVX2 aligned loads
    for (; i + 31 < total; i += 32) {
        // Load 32 bytes at once (aligned load)
        __m256i bytes = _mm256_load_si256((const __m256i*)(src + i));
        
        // Split into 8 sets of 4 bytes and process
        for (int j = 0; j < 8; j++) {
            // Extract 4 bytes
            int byte_idx = i + j * 4;
            __m128i byte_vec_32 = _mm_cvtsi32_si128(*(const int*)(src + byte_idx));
            
            // Convert 4 bytes to 4 floats
            __m128i expanded = _mm_cvtepu8_epi32(byte_vec_32);
            __m128 float_vec = _mm_cvtepi32_ps(expanded);
            
            // Apply transformation
            float_vec = _mm_add_ps(float_vec, _mm_set1_ps(0.5f));
            float_vec = _mm_mul_ps(float_vec, _mm_set1_ps(1.0f / 256.0f));
            
            _mm_storeu_ps(output + byte_idx, float_vec);
        }
    }
    
    // Process remaining bytes (up to 31)
    for (; i + 3 < total; i += 4) {
        __m128i byte_vec = _mm_cvtsi32_si128(*(const int*)(src + i));
        byte_vec = _mm_unpacklo_epi8(byte_vec, _mm_setzero_si128());
        byte_vec = _mm_unpacklo_epi16(byte_vec, _mm_setzero_si128());
        __m128 float_vec = _mm_cvtepi32_ps(byte_vec);
        
        float_vec = _mm_add_ps(float_vec, _mm_set1_ps(0.5f));
        float_vec = _mm_mul_ps(float_vec, _mm_set1_ps(1.0f / 256.0f));
        
        _mm_storeu_ps(output + i, float_vec);
    }
    
    // Process tail (0-3 elements) scalar
    for (; i < total; i++) {
        output[i] = (src[i] + 0.5f) / 256.0f;
    }
}

// Fast constant buffer update using SSE
__forceinline void UpdateConstantBuffer_SSE(ID3D11DeviceContext* context, 
                                            ID3D11Buffer* buffer, 
                                            const int* data) {
    D3D11_MAPPED_SUBRESOURCE resource;
    context->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
    
    // Copy 16 bytes at once using SSE
    __m128i vec = _mm_loadu_si128((const __m128i*)data);
    _mm_storeu_si128((__m128i*)resource.pData, vec);
    
    context->Unmap(buffer, 0);
}

const unsigned char COverlayContext_Present_bytes[] = {
	0x48, 0x89, 0x5c, 0x24, 0x08, 0x48, 0x89, 0x74, 0x24, 0x10, 0x57, 0x48, 0x83, 0xec, 0x40, 0x48, 0x8b, 0xb1, 0x20,
	0x2c, 0x00, 0x00, 0x45, 0x8b, 0xd0, 0x48, 0x8b, 0xfa, 0x48, 0x8b, 0xd9, 0x48, 0x85, 0xf6, 0x0f, 0x85
};
const int IOverlaySwapChain_IDXGISwapChain_offset = -0x118;

const unsigned char COverlayContext_IsCandidateDirectFlipCompatbile_bytes[] = {
	0x48, 0x89, 0x7c, 0x24, 0x20, 0x55, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x48, 0x8b, 0xec, 0x48, 0x83,
	0xec, 0x40
};
const unsigned char COverlayContext_OverlaysEnabled_bytes[] = {
	0x75, 0x04, 0x32, 0xc0, 0xc3, 0xcc, 0x83, 0x79, 0x30, 0x01, 0x0f, 0x97, 0xc0, 0xc3
};

const int COverlayContext_DeviceClipBox_offset = -0x120;

const int IOverlaySwapChain_HardwareProtected_offset = -0xbc;

const unsigned char COverlayContext_Present_bytes_w11[] = {
	0x40, 0x53, 0x55, 0x56, 0x57, 0x41, 0x56, 0x41, 0x57, 0x48, 0x81, 0xec, 0x88, 0x00, 0x00, 0x00, 0x48, 0x8b, 0x05,
	'?', '?', '?', '?', 0x48, 0x33, 0xc4, 0x48, 0x89, 0x44, 0x24, 0x78, 0x48
};
const int IOverlaySwapChain_IDXGISwapChain_offset_w11 = 0xE0;

const unsigned char COverlayContext_IsCandidateDirectFlipCompatbile_bytes_w11[] = {
	0x40, 0x55, 0x53, 0x56, 0x57, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x48, 0x8b, 0xec, 0x48, 0x83, 0xec,
	0x68, 0x48,
};

const unsigned char COverlayContext_OverlaysEnabled_bytes_w11[] = {
	0x83, 0x3D, '?', '?', '?', '?', '?', 0x75, 0x04
};

int COverlayContext_DeviceClipBox_offset_w11 = 0x466C;

const int IOverlaySwapChain_HardwareProtected_offset_w11 = -0x144;

bool isWindows11;

bool aob_match_inverse(const void* buf1, const void* mask, const int buf_len)
{
	for (int i = 0; i < buf_len; ++i)
	{
		if (((unsigned char*)buf1)[i] != ((unsigned char*)mask)[i] && ((unsigned char*)mask)[i] != '?')
		{
			return true;
		}
	}
	return false;
}

char shaders[] = R"(
    struct VS_INPUT {
	float2 pos : POSITION;
	float2 tex : TEXCOORD;
};

struct VS_OUTPUT {
	float4 pos : SV_POSITION;
	float2 tex : TEXCOORD;
};

Texture2D backBufferTex : register(t0);
Texture3D lutTex : register(t1);
SamplerState smp : register(s0);

Texture2D noiseTex : register(t2);
SamplerState noiseSmp : register(s1);

int lutSize : register(b0);
bool hdr : register(b1);

static float3x3 scrgb_to_bt2100 = {
2939026994.L / 585553224375.L, 9255011753.L / 3513319346250.L,   173911579.L / 501902763750.L,
  76515593.L / 138420033750.L, 6109575001.L / 830520202500.L,    75493061.L / 830520202500.L,
  12225392.L / 93230009375.L, 1772384008.L / 2517210253125.L, 18035212433.L / 2517210253125.L,
};

static float3x3 bt2100_to_scrgb = {
 348196442125.L / 1677558947.L, -123225331250.L / 1677558947.L,  -15276242500.L / 1677558947.L,
-579752563250.L / 37238079773.L, 5273377093000.L / 37238079773.L,  -38864558125.L / 37238079773.L,
 -12183628000.L / 5369968309.L, -472592308000.L / 37589778163.L, 5256599974375.L / 37589778163.L,
};

static float m1 = 1305 / 8192.;
static float m2 = 2523 / 32.;
static float c1 = 107 / 128.;
static float c2 = 2413 / 128.;
static float c3 = 2392 / 128.;

// Use hardware trilinear filtering
float3 SampleLut(float3 index) {
	float3 tex = (index + 0.5) / lutSize;
	return lutTex.Sample(smp, tex).rgb;
}

float3 LutTransform(float3 rgb) {
	float3 lutIndex = rgb * (lutSize - 1);
	return SampleLut(lutIndex);
}

float3 pq_eotf(float3 e) {
	return pow(max((pow(e, 1 / m2) - c1), 0) / (c2 - c3 * pow(e, 1 / m2)), 1 / m1);
}

float3 pq_inv_eotf(float3 y) {
	return pow((c1 + c2 * pow(y, m1)) / (1 + c3 * pow(y, m1)), m2);
}

float3 OrderedDither(float3 rgb, float2 pos) {
	float3 low = floor(rgb * 255) / 255;
	float3 high = low + 1.0 / 255;

	float3 rgb_linear = pow(rgb,)" STRINGIFY(DITHER_GAMMA) R"();
	float3 low_linear = pow(low,)" STRINGIFY(DITHER_GAMMA) R"();
	float3 high_linear = pow(high,)" STRINGIFY(DITHER_GAMMA) R"();

	float noise = noiseTex.Sample(noiseSmp, pos / )" STRINGIFY(NOISE_SIZE) R"().x;
	float3 threshold = lerp(low_linear, high_linear, noise);

	return lerp(low, high, rgb_linear > threshold);
}

VS_OUTPUT VS(VS_INPUT input) {
	VS_OUTPUT output;
	output.pos = float4(input.pos, 0, 1);
	output.tex = input.tex;
	return output;
}

float4 PS(VS_OUTPUT input) : SV_TARGET{
	float3 sample = backBufferTex.Sample(smp, input.tex).rgb;

	if (hdr) {
		float3 hdr10_sample = pq_inv_eotf(saturate(mul(scrgb_to_bt2100, sample)));

		float3 hdr10_res = LutTransform(hdr10_sample);

		float3 scrgb_res = mul(bt2100_to_scrgb, pq_eotf(hdr10_res));

		return float4(scrgb_res, 1);
	}
	else {
		float3 res = LutTransform(sample);

		res = OrderedDither(res, input.pos.xy);

		return float4(res, 1);
	}
}
)";

// Initialize all global variables
ID3D11Device* device = nullptr;
ID3D11DeviceContext* deviceContext = nullptr;
ID3D11VertexShader* vertexShader = nullptr;
ID3D11PixelShader* pixelShader = nullptr;
ID3D11InputLayout* inputLayout = nullptr;

ID3D11Buffer* vertexBuffer = nullptr;
UINT numVerts = 0;
UINT stride = 0;
UINT offset = 0;

D3D11_TEXTURE2D_DESC backBufferDesc = {};
D3D11_TEXTURE2D_DESC textureDesc[2] = {};

ID3D11SamplerState* samplerState = nullptr;
ID3D11Texture2D* texture[2] = { nullptr, nullptr };
ID3D11ShaderResourceView* textureView[2] = { nullptr, nullptr };

ID3D11SamplerState* noiseSamplerState = nullptr;
ID3D11ShaderResourceView* noiseTextureView = nullptr;

ID3D11Buffer* constantBuffer = nullptr;

// Cache for state changes to avoid redundant calls
struct CachedState {
	ID3D11ShaderResourceView* lastBackBufferView = nullptr;
	ID3D11ShaderResourceView* lastLutView = nullptr;
	ID3D11SamplerState* lastSamplerState = nullptr;
	ID3D11SamplerState* lastNoiseSamplerState = nullptr;
	int lastLutSize = 0;
	bool lastHdrState = false;
	int lastConstantBufferData[4] = { 0, 0, 0, 0 };
} g_cachedState;

struct lutData
{
	int left;
	int top;
	int size;
	bool isHdr;
	ID3D11ShaderResourceView* textureView;
	float* rawLut;

	// Initialize members to prevent uninitialized use
	lutData() : left(0), top(0), size(0), isHdr(false), textureView(nullptr), rawLut(nullptr) {}
};

void DrawRectangle(struct tagRECT* rect, int index)
{
	float width = (float)backBufferDesc.Width;
	float height = (float)backBufferDesc.Height;

	float screenLeft = (float)rect->left / width;
	float screenTop = (float)rect->top / height;
	float screenRight = (float)rect->right / width;
	float screenBottom = (float)rect->bottom / height;

	float left = screenLeft * 2 - 1;
	float top = screenTop * -2 + 1;
	float right = screenRight * 2 - 1;
	float bottom = screenBottom * -2 + 1;

	width = (float)textureDesc[index].Width;
	height = (float)textureDesc[index].Height;
	float texLeft = (float)rect->left / width;
	float texTop = (float)rect->top / height;
	float texRight = (float)rect->right / width;
	float texBottom = (float)rect->bottom / height;

	float vertexData[] = {
		left, bottom, texLeft, texBottom,
		left, top, texLeft, texTop,
		right, bottom, texRight, texBottom,
		right, top, texRight, texTop
	};

	D3D11_MAPPED_SUBRESOURCE resource;
	EXECUTE_WITH_LOG(deviceContext->Map(vertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource))
		memcpy(resource.pData, vertexData, stride * numVerts);
	deviceContext->Unmap(vertexBuffer, 0);

	deviceContext->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);

	deviceContext->Draw(numVerts, 0);
}

int numLuts = 0;

lutData* luts = nullptr;

bool ParseLUT(lutData* lut, char* filename)
{
	FILE* file = nullptr;
	if (fopen_s(&file, filename, "r") != 0 || file == NULL) return false;

	char line[512];  // Increased buffer size
	unsigned int lutSize;

	while (1)
	{
		if (!fgets(line, sizeof(line), file))
		{
			fclose(file);
			return false;
		}
		if (sscanf_s(line, "LUT_3D_SIZE %u", &lutSize) == 1)
		{
			break;
		}
	}

	float* rawLut = (float*)malloc(lutSize * lutSize * lutSize * 4 * sizeof(float));
	if (rawLut == NULL) {
		fclose(file);
		return false;
	}

	for (unsigned int b = 0; b < lutSize; b++)
	{
		for (unsigned int g = 0; g < lutSize; g++)
		{
			for (unsigned int r = 0; r < lutSize; r++)
			{
				while (1)
				{
					if (!fgets(line, sizeof(line), file))
					{
						fclose(file);
						free(rawLut);
						return false;
					}
					if (line[0] <= '9' && line[0] != '#' && line[0] != '\n')
					{
						float red, green, blue;

						if (sscanf_s(line, "%f %f %f", &red, &green, &blue) != 3)
						{
							fclose(file);
							free(rawLut);
							return false;
						}
						unsigned int index = (b * lutSize * lutSize + g * lutSize + r) * 4;
						rawLut[index + 0] = red;
						rawLut[index + 1] = green;
						rawLut[index + 2] = blue;
						rawLut[index + 3] = 1.0f;

						break;
					}
				}
			}
		}
	}
	fclose(file);
	lut->size = (int)lutSize;
	lut->rawLut = rawLut;
	return true;
}

bool AddLUTs(char* folder)
{
	WIN32_FIND_DATAA findData;

	char path[MAX_PATH];
	// Safer path construction
	if (strcpy_s(path, MAX_PATH, folder) != 0) return false;
	if (strcat_s(path, MAX_PATH, "\\*") != 0) return false;
	
	HANDLE hFind = FindFirstFileA(path, &findData);
	if (hFind == INVALID_HANDLE_VALUE) return false;
	
	do
	{
		if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
		{
			char filePath[MAX_PATH];
			char* fileName = findData.cFileName;

			// Safer file path construction
			if (strcpy_s(filePath, MAX_PATH, folder) != 0) continue;
			if (strcat_s(filePath, MAX_PATH, "\\") != 0) continue;
			if (strcat_s(filePath, MAX_PATH, fileName) != 0) continue;

			// Better memory allocation with exception handling
			lutData* newLuts = (lutData*)realloc(luts, (numLuts + 1) * sizeof(lutData));
			if (newLuts == NULL) {
				FindClose(hFind);
				return false;
			}
			luts = newLuts;

			// Initialize the new element properly
			memset(&luts[numLuts], 0, sizeof(lutData));
			
			lutData* lut = &luts[numLuts];
			
			if (sscanf_s(findData.cFileName, "%d_%d", &lut->left, &lut->top) == 2)
			{
				lut->isHdr = strstr(fileName, "hdr") != NULL;
				lut->textureView = NULL;
				if (!ParseLUT(lut, filePath))
				{
					#if DEBUG_MODE
					char errorMsg[256];
					sprintf_s(errorMsg, sizeof(errorMsg), "LUT could not be parsed: %s", filePath);
					MESSAGE_BOX_DBG(errorMsg, MB_OK | MB_ICONWARNING)
					#endif
					FindClose(hFind);
					return false;
				}
				numLuts++;
			}
		}
	} while (FindNextFileA(hFind, &findData) != 0);
	
	FindClose(hFind);
	return true;
}

int numLutTargets = 0;
void** lutTargets = nullptr;

bool IsLUTActive(void* target)
{
	for (int i = 0; i < numLutTargets; i++)
	{
		if (lutTargets[i] == target)
		{
			return true;
		}
	}
	return false;
}

void SetLUTActive(void* target)
{
	if (!IsLUTActive(target))
	{
		// Check realloc result before assigning
		void** newLutTargets = (void**)realloc(lutTargets, (numLutTargets + 1) * sizeof(void*));
		if (newLutTargets == NULL && numLutTargets + 1 > 0) {
			return;
		}
		lutTargets = newLutTargets;
		lutTargets[numLutTargets++] = target;
	}
}

void UnsetLUTActive(void* target)
{
	for (int i = 0; i < numLutTargets; i++)
	{
		if (lutTargets[i] == target)
		{
			lutTargets[i] = lutTargets[--numLutTargets];
			// Check realloc result before assigning
			void** newLutTargets = (void**)realloc(lutTargets, numLutTargets * sizeof(void*));
			if (newLutTargets == NULL && numLutTargets > 0) {
				// Keep old allocation if resize fails
				return;
			}
			lutTargets = newLutTargets;
			return;
		}
	}
}

lutData* GetLUTDataFromCOverlayContext(void* context, bool hdr)
{
	int left, top;
	if (isWindows11)
	{
		float* rect = (float*)((unsigned char*)*(void**)context + COverlayContext_DeviceClipBox_offset_w11);
		left = (int)rect[0];
		top = (int)rect[1];
	}
	else
	{
		int* rect = (int*)((unsigned char*)context + COverlayContext_DeviceClipBox_offset);
		left = rect[0];
		top = rect[1];
	}

	for (int i = 0; i < numLuts; i++)
	{
		if (luts[i].left == left && luts[i].top == top && luts[i].isHdr == hdr)
		{
			return &luts[i];
		}
	}
	return NULL;
}

bool DetectWindows11()
{
    OSVERSIONINFOEXW osvi = { sizeof(osvi), 0, 0, 0, 0, {0}, 0, 0 };
    DWORDLONG const dwlConditionMask = VerSetConditionMask(
        VerSetConditionMask(
            VerSetConditionMask(
                0, VER_MAJORVERSION, VER_GREATER_EQUAL),
            VER_MINORVERSION, VER_GREATER_EQUAL),
        VER_BUILDNUMBER, VER_GREATER_EQUAL);
    
    osvi.dwMajorVersion = 10;
    osvi.dwMinorVersion = 0;
    osvi.dwBuildNumber = 22000;  // Windows 11 starts from build 22000
    
    return VerifyVersionInfoW(&osvi, VER_MAJORVERSION | VER_MINORVERSION | VER_BUILDNUMBER, dwlConditionMask) != FALSE;
}

void InitializeStuff(IDXGISwapChain* swapChain)
{
	try
	{
		EXECUTE_WITH_LOG(swapChain->GetDevice(IID_ID3D11Device, (void**)&device))
			LOG_ADDRESS("Current swapchain address is: ", swapChain)
			LOG_ONLY_ONCE("Device successfully gathered")
			LOG_ADDRESS("The device address is: ", device)

			device->GetImmediateContext(&deviceContext);
		LOG_ONLY_ONCE("Got context after device")
			LOG_ADDRESS("The Device context is located at address: ", deviceContext)
		{
			ID3DBlob* vsBlob;
			ID3DBlob* compile_error_interface;
			LOG_ONLY_ONCE(("Trying to compile vshader with this code:\n" + std::string(shaders)).c_str())
				EXECUTE_D3DCOMPILE_WITH_LOG(
					D3DCompile(shaders, sizeof shaders, NULL, NULL, NULL, "VS", "vs_5_0", 0, 0, &vsBlob, &
						compile_error_interface), compile_error_interface)

				LOG_ONLY_ONCE("Vertex shader compiled successfully")
				EXECUTE_WITH_LOG(device->CreateVertexShader(vsBlob->GetBufferPointer(),
					vsBlob->GetBufferSize(), NULL, &vertexShader))

				LOG_ONLY_ONCE("Vertex shader created successfully")
				D3D11_INPUT_ELEMENT_DESC inputElementDesc[] =
			{
				{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
				{
					"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT,
					D3D11_INPUT_PER_VERTEX_DATA, 0
				}
			};
			EXECUTE_WITH_LOG(device->CreateInputLayout(inputElementDesc, ARRAYSIZE(inputElementDesc),
				vsBlob->GetBufferPointer(),
				vsBlob->GetBufferSize(), &inputLayout))

				vsBlob->Release();
		}
		{
			ID3DBlob* psBlob;
			ID3DBlob* compile_error_interface;
			EXECUTE_D3DCOMPILE_WITH_LOG(
				D3DCompile(shaders, sizeof shaders, NULL, NULL, NULL, "PS", "ps_5_0", 0, 0, &psBlob, &
					compile_error_interface), compile_error_interface)

				LOG_ONLY_ONCE("Pixel shader compiled successfully")
				device->CreatePixelShader(psBlob->GetBufferPointer(),
					psBlob->GetBufferSize(), NULL, &pixelShader);
			psBlob->Release();
		}
		{
			stride = 4 * sizeof(float);
			numVerts = 4;
			offset = 0;

			D3D11_BUFFER_DESC vertexBufferDesc = {};
			vertexBufferDesc.ByteWidth = stride * numVerts;
			vertexBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
			vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			vertexBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			EXECUTE_WITH_LOG(device->CreateBuffer(&vertexBufferDesc, NULL, &vertexBuffer))
		}
		{
			// Use linear filtering for hardware trilinear interpolation
			D3D11_SAMPLER_DESC samplerDesc = {};
			samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
			samplerDesc.AddressU = samplerDesc.AddressV = samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
			samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
			samplerDesc.MaxAnisotropy = 1;
			samplerDesc.MinLOD = 0;
			samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

			EXECUTE_WITH_LOG(device->CreateSamplerState(&samplerDesc, &samplerState))
		}

		// Convert float LUT data to 16-bit float using AVX2 optimization
		for (int i = 0; i < numLuts; i++)
		{
			lutData* lut = &luts[i];

			// Check if rawLut is valid before using it
			if (lut->rawLut == NULL) {
				continue; // Skip invalid LUT
			}

			// Create temporary buffer for 16-bit float data
			int elementCount = lut->size * lut->size * lut->size * 4;
			uint16_t* halfData = (uint16_t*)_aligned_malloc(elementCount * sizeof(uint16_t), 32);
			if (halfData == NULL) {
				throw std::exception("Failed to allocate memory for LUT conversion");
			}

			// Use AVX2 optimized batch conversion
			FloatToHalfBatch_AVX2(lut->rawLut, halfData, elementCount);

			D3D11_TEXTURE3D_DESC desc = {};
			desc.Width = lut->size;
			desc.Height = lut->size;
			desc.Depth = lut->size;
			desc.MipLevels = 1;
			desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; // Use 16-bit float format
			desc.Usage = D3D11_USAGE_IMMUTABLE; // Use immutable for static LUTs
			desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			desc.CPUAccessFlags = 0;
			desc.MiscFlags = 0;

			D3D11_SUBRESOURCE_DATA initData;
			initData.pSysMem = halfData;
			initData.SysMemPitch = (UINT)(lut->size * 4 * sizeof(uint16_t));
			initData.SysMemSlicePitch = (UINT)(lut->size * lut->size * 4 * sizeof(uint16_t));

			ID3D11Texture3D* tex;
			EXECUTE_WITH_LOG(device->CreateTexture3D(&desc, &initData, &tex))

				// Check tex is not null before using it
				if (tex == NULL) {
					_aligned_free(halfData);
					throw std::exception("Failed to create 3D texture for LUT");
				}

			// Check CreateShaderResourceView result
			HRESULT hr = device->CreateShaderResourceView((ID3D11Resource*)tex, NULL, &luts[i].textureView);
			if (FAILED(hr) || luts[i].textureView == NULL) {
				tex->Release();
				_aligned_free(halfData);
				throw std::exception("Failed to create shader resource view for LUT");
			}

			tex->Release();
			_aligned_free(halfData);
			free(lut->rawLut);
			lut->rawLut = NULL;
		}
		{
			D3D11_SAMPLER_DESC samplerDesc = {};
			samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
			samplerDesc.AddressU = samplerDesc.AddressV = samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
			samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;

			EXECUTE_WITH_LOG(device->CreateSamplerState(&samplerDesc, &noiseSamplerState))
		}
		{
			D3D11_TEXTURE2D_DESC desc = {};
			desc.Width = NOISE_SIZE;
			desc.Height = NOISE_SIZE;
			desc.MipLevels = 1;
			desc.ArraySize = 1;
			desc.Format = DXGI_FORMAT_R32_FLOAT;
			desc.SampleDesc.Count = 1;
			desc.SampleDesc.Quality = 0;
			desc.Usage = D3D11_USAGE_IMMUTABLE;
			desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

			// Use aligned allocation for AVX2 operations (32-byte alignment)
			float* noise = (float*)_aligned_malloc(NOISE_SIZE * NOISE_SIZE * sizeof(float), 32);
			if (noise == NULL) {
				throw std::exception("Failed to allocate memory for noise texture");
			}

			// Use AVX2 optimized noise conversion
			ConvertNoiseBytesToFloat_AVX2(noiseBytes, noise);

			D3D11_SUBRESOURCE_DATA initData;
			initData.pSysMem = noise;
			initData.SysMemPitch = NOISE_SIZE * sizeof(float);

			ID3D11Texture2D* tex;
			EXECUTE_WITH_LOG(device->CreateTexture2D(&desc, &initData, &tex))
			if (tex == NULL) {
				_aligned_free(noise);
				throw std::exception("Failed to create noise texture");
			}
			
			HRESULT hr = device->CreateShaderResourceView((ID3D11Resource*)tex, NULL, &noiseTextureView);
			if (FAILED(hr) || noiseTextureView == NULL) {
				tex->Release();
				_aligned_free(noise);
				throw std::exception("Failed to create noise texture view");
			}
				
			tex->Release();
			_aligned_free(noise);
		}
		{
			D3D11_BUFFER_DESC constantBufferDesc = {};
			constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			constantBufferDesc.ByteWidth = 16; // For int + bool (4 + 4 bytes, padded to 16)
			constantBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
			constantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

			EXECUTE_WITH_LOG(device->CreateBuffer(&constantBufferDesc, NULL, &constantBuffer))
				LOG_ONLY_ONCE("Final buffer created in InitializeStuff")
		}

		// Set samplers once during initialization
		deviceContext->PSSetSamplers(0, 1, &samplerState);
		deviceContext->PSSetSamplers(1, 1, &noiseSamplerState);

		// Initialize cache state
		g_cachedState.lastSamplerState = samplerState;
		g_cachedState.lastNoiseSamplerState = noiseSamplerState;
	}
	catch (std::exception& ex)
	{
		std::stringstream ex_message;
		ex_message << "Exception caught at line " << __LINE__ << ": " << ex.what() << std::endl;
		LOG_ONLY_ONCE(ex_message.str().c_str())
			throw;
	}
	catch (...)
	{
		std::stringstream ex_message;
		ex_message << "Exception caught at line " << __LINE__ << ": " << std::endl;
		LOG_ONLY_ONCE(ex_message.str().c_str())
			throw;
	}
}

void UninitializeStuff()
{
	RELEASE_IF_NOT_NULL(device)
	RELEASE_IF_NOT_NULL(deviceContext)
	RELEASE_IF_NOT_NULL(vertexShader)
	RELEASE_IF_NOT_NULL(pixelShader)
	RELEASE_IF_NOT_NULL(inputLayout)
	RELEASE_IF_NOT_NULL(vertexBuffer)
	RELEASE_IF_NOT_NULL(samplerState)
	
	for (int i = 0; i < 2; i++)
	{
		RELEASE_IF_NOT_NULL(texture[i])
		RELEASE_IF_NOT_NULL(textureView[i])
	}
	
	RELEASE_IF_NOT_NULL(noiseSamplerState)
	RELEASE_IF_NOT_NULL(noiseTextureView)
	RELEASE_IF_NOT_NULL(constantBuffer)

	// Check rawLut is not null before freeing
	for (int i = 0; i < numLuts; i++)
	{
		if (luts[i].rawLut != NULL) {
			free(luts[i].rawLut);
			luts[i].rawLut = NULL;
		}
		RELEASE_IF_NOT_NULL(luts[i].textureView)
	}
	
	if (luts != NULL) {
		free(luts);
		luts = NULL;
		numLuts = 0;
	}
	
	if (lutTargets != NULL) {
		free(lutTargets);
		lutTargets = NULL;
		numLutTargets = 0;
	}

	// Reset cache state
	g_cachedState = CachedState();
}

bool ApplyLUT(void* cOverlayContext, IDXGISwapChain* swapChain, struct tagRECT* rects, int numRects)
{
	try
	{
		if (!device)
		{
			LOG_ONLY_ONCE("Initializing stuff in ApplyLUT")
				InitializeStuff(swapChain);
		}
		LOG_ONLY_ONCE("Init done, continuing with LUT application")

			ID3D11Texture2D* backBuffer;
		ID3D11RenderTargetView* renderTargetView;

		EXECUTE_WITH_LOG(swapChain->GetBuffer(0, IID_ID3D11Texture2D, (void**)&backBuffer))

			// Check backBuffer is not null
			if (backBuffer == NULL) {
				return false;
			}

		D3D11_TEXTURE2D_DESC newBackBufferDesc;
		backBuffer->GetDesc(&newBackBufferDesc);

		int index = -1;
		if (newBackBufferDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM)
		{
			index = 0;
		}
		else if (newBackBufferDesc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT)
		{
			index = 1;
		}

		lutData* lut;
		if (index == -1 || !(lut = GetLUTDataFromCOverlayContext(cOverlayContext, index == 1)))
		{
			backBuffer->Release();
			return false;
		}

		D3D11_TEXTURE2D_DESC oldTextureDesc = textureDesc[index];
		if (newBackBufferDesc.Width > oldTextureDesc.Width || newBackBufferDesc.Height > oldTextureDesc.Height)
		{
			if (texture[index] != NULL)
			{
				texture[index]->Release();
				textureView[index]->Release();
			}

			UINT newWidth = max(newBackBufferDesc.Width, oldTextureDesc.Width);
			UINT newHeight = max(newBackBufferDesc.Height, oldTextureDesc.Height);

			D3D11_TEXTURE2D_DESC newTextureDesc;

			newTextureDesc = newBackBufferDesc;
			newTextureDesc.Width = newWidth;
			newTextureDesc.Height = newHeight;
			newTextureDesc.Usage = D3D11_USAGE_DEFAULT;
			newTextureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			newTextureDesc.CPUAccessFlags = 0;
			newTextureDesc.MiscFlags = 0;

			textureDesc[index] = newTextureDesc;

			EXECUTE_WITH_LOG(device->CreateTexture2D(&textureDesc[index], NULL, &texture[index]))

				// Check texture creation succeeded
				if (texture[index] == NULL) {
					backBuffer->Release();
					return false;
				}

			EXECUTE_WITH_LOG(
				device->CreateShaderResourceView((ID3D11Resource*)texture[index], NULL, &textureView[index]))

				// Check shader resource view creation succeeded
				if (textureView[index] == NULL) {
					texture[index]->Release();
					texture[index] = NULL;
					backBuffer->Release();
					return false;
				}

			// Reset cache since texture changed
			g_cachedState.lastBackBufferView = nullptr;
		}

		backBufferDesc = newBackBufferDesc;

		EXECUTE_WITH_LOG(device->CreateRenderTargetView((ID3D11Resource*)backBuffer, NULL, &renderTargetView))

			// Check render target view creation succeeded
			if (renderTargetView == NULL) {
				backBuffer->Release();
				return false;
			}

		const D3D11_VIEWPORT d3d11_viewport(0.0f, 0.0f, (float)backBufferDesc.Width, (float)backBufferDesc.Height, 0.0f, 1.0f);
		deviceContext->RSSetViewports(1, &d3d11_viewport);

		deviceContext->OMSetRenderTargets(1, &renderTargetView, NULL);
		renderTargetView->Release();

		deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
		deviceContext->IASetInputLayout(inputLayout);

		deviceContext->VSSetShader(vertexShader, NULL, 0);
		deviceContext->PSSetShader(pixelShader, NULL, 0);

		// Optimize: Only set resources if they changed
		if (g_cachedState.lastBackBufferView != textureView[index])
		{
			deviceContext->PSSetShaderResources(0, 1, &textureView[index]);
			g_cachedState.lastBackBufferView = textureView[index];
		}

		if (g_cachedState.lastLutView != lut->textureView)
		{
			deviceContext->PSSetShaderResources(1, 1, &lut->textureView);
			g_cachedState.lastLutView = lut->textureView;
		}

		// Set noise texture (usually doesn't change)
		deviceContext->PSSetShaderResources(2, 1, &noiseTextureView);

		// Update constant buffer only if values changed
		int constantData[4] = { lut->size, (index == 1) ? 1 : 0, 0, 0 };

		bool constantBufferNeedsUpdate =
			(g_cachedState.lastLutSize != lut->size) ||
			(g_cachedState.lastHdrState != (index == 1));

		if (constantBufferNeedsUpdate)
		{
			// Use SSE optimized constant buffer update
			UpdateConstantBuffer_SSE(deviceContext, constantBuffer, constantData);
			
			g_cachedState.lastLutSize = lut->size;
			g_cachedState.lastHdrState = (index == 1);
		}

		deviceContext->PSSetConstantBuffers(0, 1, &constantBuffer);

		for (int i = 0; i < numRects; i++)
		{
			D3D11_BOX sourceRegion;
			sourceRegion.left = rects[i].left;
			sourceRegion.right = rects[i].right;
			sourceRegion.top = rects[i].top;
			sourceRegion.bottom = rects[i].bottom;
			sourceRegion.front = 0;
			sourceRegion.back = 1;

			// Check texture is valid before using it
			if (texture[index] != NULL) {
				deviceContext->CopySubresourceRegion((ID3D11Resource*)texture[index], 0, rects[i].left,
					rects[i].top, 0, (ID3D11Resource*)backBuffer, 0, &sourceRegion);
				DrawRectangle(&rects[i], index);
			}
		}

		backBuffer->Release();
		return true;
	}
	catch (std::exception& ex)
	{
		std::stringstream ex_message;
		ex_message << "Exception caught at line " << __LINE__ << ": " << ex.what() << std::endl;
		LOG_ONLY_ONCE(ex_message.str().c_str())
			return false;
	}
	catch (...)
	{
		std::stringstream ex_message;
		ex_message << "Exception caught at line " << __LINE__ << std::endl;
		LOG_ONLY_ONCE(ex_message.str().c_str())
			return false;
	}
}

typedef struct rectVec
{
	struct tagRECT* start;
	struct tagRECT* end;
	struct tagRECT* cap;
} rectVec;

typedef long (COverlayContext_Present_t)(void*, void*, unsigned int, rectVec*, unsigned int, bool);

// Initialize MinHook function pointers to nullptr
COverlayContext_Present_t* COverlayContext_Present_orig = nullptr;
COverlayContext_Present_t* COverlayContext_Present_real_orig = nullptr;

long COverlayContext_Present_hook(void* self, void* overlaySwapChain, unsigned int a3, rectVec* rectVec,
	unsigned int a5, bool a6)
{
	if (_ReturnAddress() < (void*)COverlayContext_Present_real_orig)
	{
		LOG_ONLY_ONCE("I am inside COverlayContext::Present hook inside the main if condition")

			if (isWindows11 && *((bool*)overlaySwapChain + IOverlaySwapChain_HardwareProtected_offset_w11) ||
				!isWindows11 && *((bool*)overlaySwapChain + IOverlaySwapChain_HardwareProtected_offset))
			{
				std::stringstream hw_protection_message;
				hw_protection_message << "I'm inside the Hardware protection condition - 0x" << std::hex << (bool*)
					overlaySwapChain + IOverlaySwapChain_HardwareProtected_offset_w11 << " - value: 0x" << *((bool*)
						overlaySwapChain + IOverlaySwapChain_HardwareProtected_offset_w11);
				LOG_ONLY_ONCE(hw_protection_message.str().c_str())
					UnsetLUTActive(self);
			}
			else
			{
				std::stringstream hw_protection_message;
				hw_protection_message << "I'm outside the Hardware protection condition - 0x" << std::hex << (bool*)
					overlaySwapChain + IOverlaySwapChain_HardwareProtected_offset_w11 << " - value: 0x" << *((bool*)
						overlaySwapChain + IOverlaySwapChain_HardwareProtected_offset_w11);
				LOG_ONLY_ONCE(hw_protection_message.str().c_str())

					IDXGISwapChain* swapChain;
				if (isWindows11)
				{
					LOG_ONLY_ONCE("Gathering IDXGISwapChain pointer")
						int sub_from_legacy_swapchain = *(int*)((unsigned char*)overlaySwapChain - 4);
					void* real_overlay_swap_chain = (unsigned char*)overlaySwapChain - sub_from_legacy_swapchain -
						0x1b0;
					swapChain = *(IDXGISwapChain**)((unsigned char*)real_overlay_swap_chain +
						IOverlaySwapChain_IDXGISwapChain_offset_w11);
				}
				else
				{
					swapChain = *(IDXGISwapChain**)((unsigned char*)overlaySwapChain +
						IOverlaySwapChain_IDXGISwapChain_offset);
				}

				if (ApplyLUT(self, swapChain, rectVec->start, (int)(rectVec->end - rectVec->start)))
				{
					LOG_ONLY_ONCE("Setting LUTactive")
						SetLUTActive(self);
				}
				else
				{
					LOG_ONLY_ONCE("Un-setting LUTactive")
						UnsetLUTActive(self);
				}
			}
	}

	return COverlayContext_Present_orig(self, overlaySwapChain, a3, rectVec, a5, a6);
}

typedef bool (COverlayContext_IsCandidateDirectFlipCompatbile_t)(void*, void*, void*, void*, int, unsigned int, bool,
	bool);

COverlayContext_IsCandidateDirectFlipCompatbile_t* COverlayContext_IsCandidateDirectFlipCompatbile_orig = nullptr;

bool COverlayContext_IsCandidateDirectFlipCompatbile_hook(void* self, void* a2, void* a3, void* a4, int a5,
	unsigned int a6, bool a7, bool a8)
{
	if (IsLUTActive(self))
	{
		return false;
	}
	return COverlayContext_IsCandidateDirectFlipCompatbile_orig(self, a2, a3, a4, a5, a6, a7, a8);
}

typedef bool (COverlayContext_OverlaysEnabled_t)(void*);

COverlayContext_OverlaysEnabled_t* COverlayContext_OverlaysEnabled_orig = nullptr;

bool COverlayContext_OverlaysEnabled_hook(void* self)
{
	if (IsLUTActive(self))
	{
		LOG_ONLY_ONCE("LUT ACTIVE FALSE in overlaysEnabled")
			return false;
	}
	return COverlayContext_OverlaysEnabled_orig(self);
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID lpReserved)
{
	switch (fdwReason)
	{
	case DLL_PROCESS_ATTACH:
	{
		HMODULE dwmcore = GetModuleHandle(L"dwmcore.dll");
		MODULEINFO moduleInfo;
		GetModuleInformation(GetCurrentProcess(), dwmcore, &moduleInfo, sizeof moduleInfo);

		// Use simple Windows version detection
		isWindows11 = DetectWindows11();

		MESSAGE_BOX_DBG("DWM LUT ATTACH", MB_OK)

			if (isWindows11)
			{
				MESSAGE_BOX_DBG("DETECTED WINDOWS 11 OS", MB_OK)

					for (size_t i = 0; i <= moduleInfo.SizeOfImage - sizeof COverlayContext_OverlaysEnabled_bytes_w11; i++)
					{
						unsigned char* address = (unsigned char*)dwmcore + i;
						if (!COverlayContext_Present_orig && sizeof COverlayContext_Present_bytes_w11 <= moduleInfo.
							SizeOfImage - i && !aob_match_inverse(address, COverlayContext_Present_bytes_w11,
								sizeof COverlayContext_Present_bytes_w11))
						{
							MESSAGE_BOX_DBG("DETECTED COverlayContextPresent address", MB_OK)

								COverlayContext_Present_orig = (COverlayContext_Present_t*)address;
							COverlayContext_Present_real_orig = COverlayContext_Present_orig;
						}
						else if (!COverlayContext_IsCandidateDirectFlipCompatbile_orig && sizeof
							COverlayContext_IsCandidateDirectFlipCompatbile_bytes_w11 <= moduleInfo.SizeOfImage - i && !
							aob_match_inverse(
								address, COverlayContext_IsCandidateDirectFlipCompatbile_bytes_w11,
								sizeof COverlayContext_IsCandidateDirectFlipCompatbile_bytes_w11))
						{
							COverlayContext_IsCandidateDirectFlipCompatbile_orig = (
								COverlayContext_IsCandidateDirectFlipCompatbile_t*)address;
						}
						else if (!COverlayContext_OverlaysEnabled_orig && sizeof COverlayContext_OverlaysEnabled_bytes_w11
							<= moduleInfo.SizeOfImage - i && !aob_match_inverse(
								address, COverlayContext_OverlaysEnabled_bytes_w11,
								sizeof COverlayContext_OverlaysEnabled_bytes_w11))
						{
							COverlayContext_OverlaysEnabled_orig = (COverlayContext_OverlaysEnabled_t*)address;
						}
						if (COverlayContext_Present_orig && COverlayContext_IsCandidateDirectFlipCompatbile_orig &&
							COverlayContext_OverlaysEnabled_orig)
						{
							MESSAGE_BOX_DBG("All addresses successfully retrieved", MB_OK)

								break;
						}
					}

				DWORD rev;
				DWORD revSize = sizeof(rev);
				RegGetValueA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "UBR", RRF_RT_DWORD,
					NULL, &rev, &revSize);

				if (rev >= 706)
				{
					MESSAGE_BOX_DBG("Detected recent Windows OS", MB_OK)
				}
			}
			else
			{
				for (size_t i = 0; i <= moduleInfo.SizeOfImage - sizeof(COverlayContext_Present_bytes); i++)
				{
					unsigned char* address = (unsigned char*)dwmcore + i;
					if (!COverlayContext_Present_orig && !memcmp(address, COverlayContext_Present_bytes,
						sizeof(COverlayContext_Present_bytes)))
					{
						COverlayContext_Present_orig = (COverlayContext_Present_t*)address;
						COverlayContext_Present_real_orig = COverlayContext_Present_orig;
					}
					else if (!COverlayContext_IsCandidateDirectFlipCompatbile_orig && !memcmp(
						address, COverlayContext_IsCandidateDirectFlipCompatbile_bytes,
						sizeof(COverlayContext_IsCandidateDirectFlipCompatbile_bytes)))
					{
						static int found = 0;
						found++;
						if (found == 2)
						{
							COverlayContext_IsCandidateDirectFlipCompatbile_orig = (
								COverlayContext_IsCandidateDirectFlipCompatbile_t*)(address - 0xa);
						}
					}
					else if (!COverlayContext_OverlaysEnabled_orig && !memcmp(
						address, COverlayContext_OverlaysEnabled_bytes, sizeof(COverlayContext_OverlaysEnabled_bytes)))
					{
						COverlayContext_OverlaysEnabled_orig = (COverlayContext_OverlaysEnabled_t*)(address - 0x7);
					}
					if (COverlayContext_Present_orig && COverlayContext_IsCandidateDirectFlipCompatbile_orig &&
						COverlayContext_OverlaysEnabled_orig)
					{
						break;
					}
				}
			}

		char lutFolderPath[MAX_PATH];
		ExpandEnvironmentStringsA(LUT_FOLDER, lutFolderPath, sizeof(lutFolderPath));
		if (!AddLUTs(lutFolderPath))
		{
			return FALSE;
		}
		char variable_message_states[300];
		sprintf_s(variable_message_states, sizeof(variable_message_states), "Current variable states: COverlayContext::Present - %p\t"
			"COverlayContext::IsCandidateDirectFlipCompatible - %p\tCOverlayContext::OverlaysEnabled - %p",
			COverlayContext_Present_orig,
			COverlayContext_IsCandidateDirectFlipCompatbile_orig, COverlayContext_OverlaysEnabled_orig);

		MESSAGE_BOX_DBG(variable_message_states, MB_OK)

			if (COverlayContext_Present_orig && COverlayContext_IsCandidateDirectFlipCompatbile_orig &&
				COverlayContext_OverlaysEnabled_orig && numLuts != 0)

			{
				// Check MinHook initialization
				if (MH_Initialize() != MH_OK)
				{
					MESSAGE_BOX_DBG("MinHook initialization failed", MB_OK);
					return FALSE;
				}

				MH_STATUS createHookStatus1 = MH_CreateHook((PVOID)COverlayContext_Present_orig, (PVOID)COverlayContext_Present_hook,
					(PVOID*)&COverlayContext_Present_orig);
				MH_STATUS createHookStatus2 = MH_CreateHook((PVOID)COverlayContext_IsCandidateDirectFlipCompatbile_orig,
					(PVOID)COverlayContext_IsCandidateDirectFlipCompatbile_hook,
					(PVOID*)&COverlayContext_IsCandidateDirectFlipCompatbile_orig);
				MH_STATUS createHookStatus3 = MH_CreateHook((PVOID)COverlayContext_OverlaysEnabled_orig, (PVOID)COverlayContext_OverlaysEnabled_hook,
					(PVOID*)&COverlayContext_OverlaysEnabled_orig);

				// CHECK ALL HOOK CREATIONS
				if (createHookStatus1 != MH_OK || createHookStatus2 != MH_OK || createHookStatus3 != MH_OK)
				{
					MESSAGE_BOX_DBG("Failed to create hooks", MB_OK);
					MH_Uninitialize();
					return FALSE;
				}

				MH_STATUS enableStatus = MH_EnableHook(MH_ALL_HOOKS);
				if (enableStatus != MH_OK)
				{
					MESSAGE_BOX_DBG("Failed to enable hooks", MB_OK);
					MH_Uninitialize();
					return FALSE;
				}
				
				LOG_ONLY_ONCE("DWM HOOK DLL INITIALIZATION. START LOGGING")
					MESSAGE_BOX_DBG("DWM HOOK INITIALIZATION", MB_OK)

					break;
			}
		return FALSE;
	}
	case DLL_PROCESS_DETACH:
		MH_Uninitialize();
		Sleep(100);
		UninitializeStuff();
		break;
	default:
		break;
	}
	return TRUE;
}