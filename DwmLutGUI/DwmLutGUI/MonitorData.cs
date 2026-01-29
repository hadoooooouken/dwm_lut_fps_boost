using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Windows.Input;

namespace DwmLutGUI
{
    public class MonitorData : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        public static event PropertyChangedEventHandler StaticPropertyChanged;

        private string _sdrLutPath;

        public MonitorData(string devicePath, uint sourceId, string name, string connector, string position,
            string sdrLutPath)
        {
            SdrLuts = new ObservableCollection<string>();
            DevicePath = devicePath;
            SourceId = sourceId;
            Name = name;
            Connector = connector;
            Position = position;
            SdrLutPath = sdrLutPath;
        }

        public MonitorData(string devicePath, string sdrLutPath)
        {
            SdrLuts = new ObservableCollection<string>();
            DevicePath = devicePath;
            SdrLutPath = sdrLutPath;
        }

        public string DevicePath { get; }
        public uint SourceId { get; }
        public string Name { get; }
        public string Connector { get; }
        public string Position { get; }

        public ObservableCollection<string> SdrLuts { get; set; }

        public string SdrLutPath
        {
            set
            {
                if (value == _sdrLutPath) return;
                if (value == null) return;
                if (value != "None" && !SdrLuts.Contains(value))
                    SdrLuts.Add(value);
                _sdrLutPath = value != "None" ? value : null;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(SdrLutFilename)));
                StaticPropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(SdrLutFilename)));
            }
            get => _sdrLutPath;
        }

        public string SdrLutFilename => Path.GetFileName(SdrLutPath) ?? "None";

    }
}