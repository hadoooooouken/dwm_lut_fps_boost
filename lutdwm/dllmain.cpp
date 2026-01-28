// dllmain.cpp : Defines the entry point for the DLL application.

#ifndef _DLL
#endif

#include "pch.h"

#include <io.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <DirectXMath.h>
#include <immintrin.h>
#include <intrin.h>
#include <detours/detours.h>

#pragma comment (lib, "d3d11.lib")
#pragma comment (lib, "d3dcompiler.lib")
#pragma comment (lib, "dxgi.lib")
#pragma comment (lib, "uuid.lib")
#pragma comment (lib, "dxguid.lib")
#pragma comment (lib, "detours.lib")

#pragma intrinsic(_ReturnAddress)

#define DITHER_GAMMA 2.2f
#define LUT_FOLDER "%SYSTEMROOT%\\Temp\\luts"
#define MAX_LUTS 32
#define MAX_LUT_TARGETS 64

#define RELEASE_IF_NOT_NULL(x) { if ((x) != NULL) { (x)->Release(); (x) = NULL; } }
#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)

// SSE optimized float to half conversion batch
__forceinline void FloatToHalfBatch_SSE(const float* src, uint16_t* dst, size_t count) {
    size_t i = 0;

    // Process 4 floats at a time with SSE
    for (; i + 3 < count; i += 4) {
        __m128 vf = _mm_loadu_ps(src + i);
        __m128i vh = _mm_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i*)(dst + i), vh); // Store lower 64 bits (4 half values)
    }

    // Process remaining 0-3 elements
    for (; i < count; i++) {
        __m128 vf = _mm_set_ss(src[i]);
        __m128i vh = _mm_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
        dst[i] = (uint16_t)_mm_cvtsi128_si32(vh);
    }
}


// SSE optimized noise texture conversion with aligned loads
__forceinline void ConvertNoiseBytesToFloat_SSE(float* output) {
    const uint8_t* src = GetNoiseDataPtr();
    const size_t total = static_cast<size_t>(NOISE_SIZE) * static_cast<size_t>(NOISE_SIZE);
    
    size_t i = 0;
    
    // Process 16 bytes at a time (4 sets of 4 bytes) with SSE aligned loads
    for (; i + 15 < total; i += 16) {
        __m128i bytes = _mm_load_si128((const __m128i*)(src + i));
        
        // Process each set of 4 bytes separately
        for (size_t j = 0; j < 4; j++) {
            size_t byte_idx = i + j * 4;
            // Load 4 bytes as 32-bit integer and expand to 4 floats
            __m128i byte_vec_32 = _mm_cvtsi32_si128(*(const int*)(src + byte_idx));
            
            __m128i expanded = _mm_cvtepu8_epi32(byte_vec_32);
            __m128 float_vec = _mm_cvtepi32_ps(expanded);
            
            float_vec = _mm_add_ps(float_vec, _mm_set1_ps(0.5f));
            float_vec = _mm_mul_ps(float_vec, _mm_set1_ps(1.0f / 256.0f));
            
            _mm_storeu_ps(output + byte_idx, float_vec);
        }
    }
    
    // Process remaining bytes (up to 15)
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

bool aob_match_inverse(const void* buf1, const void* mask, size_t buf_len)
{
    for (size_t i = 0; i < buf_len; ++i)
    {
        if (((const unsigned char*)buf1)[i] != ((const unsigned char*)mask)[i] && ((const unsigned char*)mask)[i] != '?')
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

// Global variables
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

struct lutData
{
    int left;
    int top;
    int size;
    bool isHdr;
    ID3D11ShaderResourceView* textureView;
    float* rawLut;

    lutData() : left(0), top(0), size(0), isHdr(false), textureView(nullptr), rawLut(nullptr) {}
};

void DrawRectangle(struct tagRECT* rect, int index)
{
    if (!device || !deviceContext) return;
    
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
    HRESULT hr = deviceContext->Map(vertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &resource);
    if (SUCCEEDED(hr)) {
        memcpy(resource.pData, vertexData, stride * numVerts);
        deviceContext->Unmap(vertexBuffer, 0);
    } else {
        return;
    }

    deviceContext->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
    deviceContext->Draw(numVerts, 0);
}

static std::vector<lutData> luts;
static std::vector<void*> lutTargets;

void InitializeVectors() {
    luts.reserve(MAX_LUTS);
    lutTargets.reserve(MAX_LUT_TARGETS);
}

bool ParseLUT(lutData* lut, char* filename)
{
    FILE* file = nullptr;
    if (fopen_s(&file, filename, "r") != 0 || file == NULL) return false;

    char line[512];
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

            if (strcpy_s(filePath, MAX_PATH, folder) != 0) continue;
            if (strcat_s(filePath, MAX_PATH, "\\") != 0) continue;
            if (strcat_s(filePath, MAX_PATH, fileName) != 0) continue;

            if (luts.size() >= luts.capacity()) {
                return false;
            }
            
            luts.emplace_back();
            lutData* lut = &luts.back();
            
            if (sscanf_s(findData.cFileName, "%d_%d", &lut->left, &lut->top) == 2)
            {
                lut->isHdr = strstr(fileName, "hdr") != NULL;
                lut->textureView = NULL;
                if (!ParseLUT(lut, filePath))
                {
                    return false;
                }
            }
        }
    } while (FindNextFileA(hFind, &findData) != 0);
    
    FindClose(hFind);
    return true;
}

bool IsLUTActive(void* target)
{
    return std::find(lutTargets.begin(), lutTargets.end(), target) != lutTargets.end();
}

void SetLUTActive(void* target)
{
    if (!IsLUTActive(target))
    {
        if (lutTargets.size() < lutTargets.capacity()) {
            lutTargets.push_back(target);
        }
    }
}

void UnsetLUTActive(void* target)
{
    auto it = std::remove(lutTargets.begin(), lutTargets.end(), target);
    if (it != lutTargets.end()) {
        lutTargets.erase(it, lutTargets.end());
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

    for (auto& lut : luts)
    {
        if (lut.left == left && lut.top == top && lut.isHdr == hdr)
        {
            return &lut;
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
    osvi.dwBuildNumber = 22000;
    
    return VerifyVersionInfoW(&osvi, VER_MAJORVERSION | VER_MINORVERSION | VER_BUILDNUMBER, dwlConditionMask) != FALSE;
}

void InitializeStuff(IDXGISwapChain* swapChain)
{
    if (!swapChain) return;
    
    HRESULT hr = swapChain->GetDevice(IID_ID3D11Device, (void**)&device);
    if (FAILED(hr) || device == nullptr) return;
    
    device->GetImmediateContext(&deviceContext);
    if (deviceContext == nullptr) {
        device->Release();
        device = nullptr;
        return;
    }
    
    {
        ID3DBlob* vsBlob = nullptr;
        ID3DBlob* compile_error_interface = nullptr;
        
        hr = D3DCompile(shaders, sizeof(shaders), NULL, NULL, NULL, "VS", "vs_5_0", 0, 0, &vsBlob, &compile_error_interface);
        if (FAILED(hr) || vsBlob == nullptr) {
            if (compile_error_interface) compile_error_interface->Release();
            return;
        }
        
        hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), NULL, &vertexShader);
        if (FAILED(hr)) {
            vsBlob->Release();
            if (compile_error_interface) compile_error_interface->Release();
            return;
        }
        
        D3D11_INPUT_ELEMENT_DESC inputElementDesc[] =
        {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {
                "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT,
                D3D11_INPUT_PER_VERTEX_DATA, 0
            }
        };
        hr = device->CreateInputLayout(inputElementDesc, ARRAYSIZE(inputElementDesc),
            vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &inputLayout);

        vsBlob->Release();
        if (compile_error_interface) compile_error_interface->Release();
        
        if (FAILED(hr)) {
            return;
        }
    }
    {
        ID3DBlob* psBlob = nullptr;
        ID3DBlob* compile_error_interface = nullptr;
        
        hr = D3DCompile(shaders, sizeof(shaders), NULL, NULL, NULL, "PS", "ps_5_0", 0, 0, &psBlob, &compile_error_interface);
        if (FAILED(hr) || psBlob == nullptr) {
            if (compile_error_interface) compile_error_interface->Release();
            return;
        }
        
        hr = device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), NULL, &pixelShader);
        psBlob->Release();
        if (compile_error_interface) compile_error_interface->Release();
        
        if (FAILED(hr)) {
            return;
        }
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

        hr = device->CreateBuffer(&vertexBufferDesc, NULL, &vertexBuffer);
        if (FAILED(hr)) {
            return;
        }
    }
    {
        D3D11_SAMPLER_DESC samplerDesc = {};
        samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        samplerDesc.AddressU = samplerDesc.AddressV = samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        samplerDesc.MaxAnisotropy = 1;
        samplerDesc.MinLOD = 0;
        samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

        hr = device->CreateSamplerState(&samplerDesc, &samplerState);
        if (FAILED(hr)) {
            return;
        }
    }

    for (auto& lut : luts)
    {
        if (lut.rawLut == NULL) {
            continue;
        }

        int elementCount = lut.size * lut.size * lut.size * 4;
        uint16_t* halfData = (uint16_t*)_aligned_malloc(elementCount * sizeof(uint16_t), 32);
        if (halfData == NULL) {
            continue;
        }

        FloatToHalfBatch_SSE(lut.rawLut, halfData, elementCount);

        D3D11_TEXTURE3D_DESC desc = {};
        desc.Width = lut.size;
        desc.Height = lut.size;
        desc.Depth = lut.size;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;

        D3D11_SUBRESOURCE_DATA initData;
        initData.pSysMem = halfData;
        initData.SysMemPitch = (UINT)(lut.size * 4 * sizeof(uint16_t));
        initData.SysMemSlicePitch = (UINT)(lut.size * lut.size * 4 * sizeof(uint16_t));

        ID3D11Texture3D* tex = nullptr;
        hr = device->CreateTexture3D(&desc, &initData, &tex);

        if (FAILED(hr) || tex == NULL) {
            _aligned_free(halfData);
            continue;
        }

        hr = device->CreateShaderResourceView((ID3D11Resource*)tex, NULL, &lut.textureView);
        if (FAILED(hr) || lut.textureView == NULL) {
            tex->Release();
            _aligned_free(halfData);
            continue;
        }

        tex->Release();
        _aligned_free(halfData);
        free(lut.rawLut);
        lut.rawLut = NULL;
    }
    {
        D3D11_SAMPLER_DESC samplerDesc = {};
        samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
        samplerDesc.AddressU = samplerDesc.AddressV = samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;

        device->CreateSamplerState(&samplerDesc, &noiseSamplerState);
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

        float* noise = (float*)_aligned_malloc(NOISE_SIZE * NOISE_SIZE * sizeof(float), 32);
        if (noise == NULL) {
            return;
        }

        ConvertNoiseBytesToFloat_SSE(noise);

        D3D11_SUBRESOURCE_DATA initData;
        initData.pSysMem = noise;
        initData.SysMemPitch = NOISE_SIZE * sizeof(float);

        ID3D11Texture2D* tex = nullptr;
        hr = device->CreateTexture2D(&desc, &initData, &tex);
        if (FAILED(hr) || tex == NULL) {
            _aligned_free(noise);
            return;
        }
        
        hr = device->CreateShaderResourceView((ID3D11Resource*)tex, NULL, &noiseTextureView);
        if (FAILED(hr) || noiseTextureView == NULL) {
            tex->Release();
            _aligned_free(noise);
            return;
        }
            
        tex->Release();
        _aligned_free(noise);
    }
    {
        D3D11_BUFFER_DESC constantBufferDesc = {};
        constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        constantBufferDesc.ByteWidth = 16;
        constantBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
        constantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        hr = device->CreateBuffer(&constantBufferDesc, NULL, &constantBuffer);
        if (FAILED(hr)) {
            return;
        }
    }

    if (deviceContext && samplerState && noiseSamplerState) {
        deviceContext->PSSetSamplers(0, 1, &samplerState);
        deviceContext->PSSetSamplers(1, 1, &noiseSamplerState);
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
    RELEASE_IF_NOT_NULL(constantBuffer)
    
    for (int i = 0; i < 2; i++)
    {
        RELEASE_IF_NOT_NULL(texture[i])
        RELEASE_IF_NOT_NULL(textureView[i])
    }
    
    RELEASE_IF_NOT_NULL(noiseSamplerState)
    RELEASE_IF_NOT_NULL(noiseTextureView)

    for (auto& lut : luts)
    {
        if (lut.rawLut != NULL) {
            free(lut.rawLut);
            lut.rawLut = NULL;
        }
        RELEASE_IF_NOT_NULL(lut.textureView)
    }
    
    luts.clear();
    lutTargets.clear();
}

bool ApplyLUT(void* cOverlayContext, IDXGISwapChain* swapChain, struct tagRECT* rects, int numRects)
{
    if (!device)
    {
        InitializeStuff(swapChain);
    }

    if (!device || !deviceContext) return false;
    
    ID3D11Texture2D* backBuffer = nullptr;
    ID3D11RenderTargetView* renderTargetView = nullptr;

    HRESULT hr = swapChain->GetBuffer(0, IID_ID3D11Texture2D, (void**)&backBuffer);
    if (FAILED(hr) || backBuffer == NULL) {
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

    lutData* lut = nullptr;
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
            texture[index] = nullptr;
            textureView[index] = nullptr;
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

        hr = device->CreateTexture2D(&textureDesc[index], NULL, &texture[index]);

        if (FAILED(hr) || texture[index] == NULL) {
            backBuffer->Release();
            return false;
        }

        hr = device->CreateShaderResourceView((ID3D11Resource*)texture[index], NULL, &textureView[index]);

        if (FAILED(hr) || textureView[index] == NULL) {
            texture[index]->Release();
            texture[index] = NULL;
            backBuffer->Release();
            return false;
        }
    }

    backBufferDesc = newBackBufferDesc;

    hr = device->CreateRenderTargetView((ID3D11Resource*)backBuffer, NULL, &renderTargetView);

    if (FAILED(hr) || renderTargetView == NULL) {
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

    deviceContext->PSSetShaderResources(0, 1, &textureView[index]);
    deviceContext->PSSetShaderResources(1, 1, &lut->textureView);
    deviceContext->PSSetSamplers(0, 1, &samplerState);

    deviceContext->PSSetShaderResources(2, 1, &noiseTextureView);
    deviceContext->PSSetSamplers(1, 1, &noiseSamplerState);

    int constantData[4] = { lut->size, (index == 1) ? 1 : 0, 0, 0 };
    UpdateConstantBuffer_SSE(deviceContext, constantBuffer, constantData);
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

        if (texture[index] != NULL) {
            deviceContext->CopySubresourceRegion((ID3D11Resource*)texture[index], 0, rects[i].left,
                rects[i].top, 0, (ID3D11Resource*)backBuffer, 0, &sourceRegion);
            DrawRectangle(&rects[i], index);
        }
    }

    backBuffer->Release();
    return true;
}

typedef struct rectVec
{
    struct tagRECT* start;
    struct tagRECT* end;
    struct tagRECT* cap;
} rectVec;

typedef long (COverlayContext_Present_t)(void*, void*, unsigned int, rectVec*, unsigned int, bool);

COverlayContext_Present_t* COverlayContext_Present_orig = nullptr;
COverlayContext_Present_t* COverlayContext_Present_real_orig = nullptr;

long COverlayContext_Present_hook(void* self, void* overlaySwapChain, unsigned int a3, rectVec* rectVec,
    unsigned int a5, bool a6)
{
    if (_ReturnAddress() < (void*)COverlayContext_Present_real_orig)
    {
        if (isWindows11 && *((bool*)overlaySwapChain + IOverlaySwapChain_HardwareProtected_offset_w11) ||
            !isWindows11 && *((bool*)overlaySwapChain + IOverlaySwapChain_HardwareProtected_offset))
        {
            UnsetLUTActive(self);
        }
        else
        {
            IDXGISwapChain* swapChain = nullptr;
            if (isWindows11)
            {
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

            if (swapChain && ApplyLUT(self, swapChain, rectVec->start, (int)(rectVec->end - rectVec->start)))
            {
                SetLUTActive(self);
            }
            else
            {
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
        DisableThreadLibraryCalls(hModule);
        
        InitializeVectors();
        
        HMODULE dwmcore = GetModuleHandle(L"dwmcore.dll");
        if (!dwmcore) return FALSE;
        
        MODULEINFO moduleInfo;
        if (!GetModuleInformation(GetCurrentProcess(), dwmcore, &moduleInfo, sizeof(moduleInfo))) {
            return FALSE;
        }

        isWindows11 = DetectWindows11();

        size_t imageSize = static_cast<size_t>(moduleInfo.SizeOfImage);

        if (isWindows11)
        {
            for (size_t i = 0; i <= imageSize - sizeof(COverlayContext_Present_bytes_w11); i++)
            {
                unsigned char* address = (unsigned char*)dwmcore + i;
                if (!COverlayContext_Present_orig && sizeof(COverlayContext_Present_bytes_w11) <= imageSize - i && 
                    !aob_match_inverse(address, COverlayContext_Present_bytes_w11,
                        sizeof(COverlayContext_Present_bytes_w11)))
                {
                    COverlayContext_Present_orig = (COverlayContext_Present_t*)address;
                    COverlayContext_Present_real_orig = COverlayContext_Present_orig;
                }
                else if (!COverlayContext_IsCandidateDirectFlipCompatbile_orig && 
                    sizeof(COverlayContext_IsCandidateDirectFlipCompatbile_bytes_w11) <= imageSize - i && 
                    !aob_match_inverse(address, COverlayContext_IsCandidateDirectFlipCompatbile_bytes_w11,
                        sizeof(COverlayContext_IsCandidateDirectFlipCompatbile_bytes_w11)))
                {
                    COverlayContext_IsCandidateDirectFlipCompatbile_orig = (
                        COverlayContext_IsCandidateDirectFlipCompatbile_t*)address;
                }
                else if (!COverlayContext_OverlaysEnabled_orig && 
                    sizeof(COverlayContext_OverlaysEnabled_bytes_w11) <= imageSize - i && 
                    !aob_match_inverse(address, COverlayContext_OverlaysEnabled_bytes_w11,
                        sizeof(COverlayContext_OverlaysEnabled_bytes_w11)))
                {
                    COverlayContext_OverlaysEnabled_orig = (COverlayContext_OverlaysEnabled_t*)address;
                }
                if (COverlayContext_Present_orig && COverlayContext_IsCandidateDirectFlipCompatbile_orig &&
                    COverlayContext_OverlaysEnabled_orig)
                {
                    break;
                }
            }
        }
        else
        {
            for (size_t i = 0; i <= imageSize - sizeof(COverlayContext_Present_bytes); i++)
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
        if (!ExpandEnvironmentStringsA(LUT_FOLDER, lutFolderPath, sizeof(lutFolderPath))) {
            return FALSE;
        }
        
        if (!AddLUTs(lutFolderPath))
        {
            return FALSE;
        }

        if (COverlayContext_Present_orig && COverlayContext_IsCandidateDirectFlipCompatbile_orig &&
            COverlayContext_OverlaysEnabled_orig && !luts.empty())
        {
            DetourTransactionBegin();
            DetourUpdateThread(GetCurrentThread());
            
            DetourAttach(&(PVOID&)COverlayContext_Present_orig, COverlayContext_Present_hook);
            DetourAttach(&(PVOID&)COverlayContext_IsCandidateDirectFlipCompatbile_orig, 
                COverlayContext_IsCandidateDirectFlipCompatbile_hook);
            DetourAttach(&(PVOID&)COverlayContext_OverlaysEnabled_orig, 
                COverlayContext_OverlaysEnabled_hook);
            
            LONG error = DetourTransactionCommit();
            if (error != NO_ERROR)
            {
                DetourTransactionBegin();
                DetourUpdateThread(GetCurrentThread());
                DetourDetach(&(PVOID&)COverlayContext_Present_orig, COverlayContext_Present_hook);
                DetourDetach(&(PVOID&)COverlayContext_IsCandidateDirectFlipCompatbile_orig, 
                    COverlayContext_IsCandidateDirectFlipCompatbile_hook);
                DetourDetach(&(PVOID&)COverlayContext_OverlaysEnabled_orig, 
                    COverlayContext_OverlaysEnabled_hook);
                DetourTransactionCommit();
                
                return FALSE;
            }
            
            break;
        }
        return FALSE;
    }
    case DLL_PROCESS_DETACH:
    {
        if (COverlayContext_Present_orig || COverlayContext_IsCandidateDirectFlipCompatbile_orig || 
            COverlayContext_OverlaysEnabled_orig)
        {
            DetourTransactionBegin();
            DetourUpdateThread(GetCurrentThread());
            
            if (COverlayContext_Present_orig)
                DetourDetach(&(PVOID&)COverlayContext_Present_orig, COverlayContext_Present_hook);
            
            if (COverlayContext_IsCandidateDirectFlipCompatbile_orig)
                DetourDetach(&(PVOID&)COverlayContext_IsCandidateDirectFlipCompatbile_orig, 
                    COverlayContext_IsCandidateDirectFlipCompatbile_hook);
            
            if (COverlayContext_OverlaysEnabled_orig)
                DetourDetach(&(PVOID&)COverlayContext_OverlaysEnabled_orig, 
                    COverlayContext_OverlaysEnabled_hook);
            
            DetourTransactionCommit();
        }
        
        Sleep(100);
        UninitializeStuff();
        break;
    }
    default:
        break;
    }
    return TRUE;
}