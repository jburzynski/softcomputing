using Encog.ML.Data;
using Encog.ML.Data.Specific;
using Encog.Neural.Thermal;
using Encog.Util;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using TrafficSigns.Encog.Neural.Thermal;
using TrafficSigns.Utils;

namespace TrafficSigns
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        protected Dictionary<string, BiPolarMLData> LoadedImages { get; set; }
        protected TrafficSignsHopfieldNetwork Network { get; set; }

        public MainWindow()
        {
            InitializeComponent();

            LoadedImages = new Dictionary<string, BiPolarMLData>();
            Network = new TrafficSignsHopfieldNetwork(ImageUtils.ImageWidth * ImageUtils.ImageHeight);
        }

        private void DrawPattern(BiPolarMLData data, Canvas canvas)
        {
            canvas.Children.Clear();
            int fieldSize = 8;

            for (int i = 0; i < data.Data.Length; i++)
            {
                int y = i / ImageUtils.ImageWidth;
                int x = i - y * ImageUtils.ImageWidth;

                Brush brush = data.Data[i] > 0 ? Brushes.White : Brushes.Black;
                var rectangle = new Rectangle
                {
                    Stroke = brush,
                    Fill = brush,
                    Width = fieldSize,
                    Height = fieldSize
                };

                canvas.Children.Add(rectangle);
                Canvas.SetLeft(rectangle, x * fieldSize);
                Canvas.SetTop(rectangle, y * fieldSize);
            }
        }

        private void EvaluatePattern(BiPolarMLData data)
        {
            int maxCycles = Int32.Parse(maxCyclesTextBox.Text);

            var currentState = new BiPolarMLData(data.Count);
            for (int i = 0; i < data.Count; i++)
            {
                currentState.SetBoolean(i, (data[i] > 0));
            }
            Network.CurrentState = currentState;

            Network.RunUntilStable(maxCycles);

            DrawPattern(Network.CurrentState, recognizedCanvas);
        }

        private void loadFileButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog {
                InitialDirectory = @"D:\PWr\_Semestr 9\Softcomputing\lab\lab1\traffic signs",
                DefaultExt = ".png",
                Multiselect = true
            };

            bool? result = dialog.ShowDialog();

            if (result == true)
            {
                foreach (string filePath in dialog.FileNames)
                {
                    string fileName = filePath.Split('\\').Last();

                    BiPolarMLData imageData = ImageUtils.GetImageData(filePath);

                    try
                    {
                        LoadedImages.Add(fileName, imageData);
                        loadedFilesListBox.Items.Add(new ListBoxItem { Content = fileName });
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("Pattern " + fileName + " is already on the list.");
                    }
                }
            }
        }

        private void testButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog { InitialDirectory = @"D:\PWr\_Semestr 9\Softcomputing\lab\lab1\traffic signs", DefaultExt = ".png" };

            bool? result = dialog.ShowDialog();

            if (result == true)
            {
                string filePath = dialog.FileName;
                BiPolarMLData imageData = ImageUtils.GetImageData(filePath);

                DrawPattern(imageData, selectedCanvas);
                EvaluatePattern(imageData);
            }
        }

        private void trainButton_Click(object sender, RoutedEventArgs e)
        {
            Network = new TrafficSignsHopfieldNetwork(ImageUtils.ImageWidth * ImageUtils.ImageHeight);

            Network.TrainDelta(LoadedImages.Values.ToList());

            //Network.TrainPseudoinverse(LoadedImages.Values.ToList());

            //Network.TrainHebbian(LoadedImages.Values.ToList());
        }

        private void deleteFilesButton_Click(object sender, RoutedEventArgs e)
        {
            LoadedImages.Clear();
            Network = null;
            loadedFilesListBox.Items.Clear();
            selectedCanvas.Children.Clear();
            recognizedCanvas.Children.Clear();
        }

        private void loadedFilesListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            ListBoxItem item = (ListBoxItem)loadedFilesListBox.SelectedItem;
            if (item != null)
            {
                DrawPattern(LoadedImages[(string)item.Content], selectedCanvas);
                recognizedCanvas.Children.Clear();
            }
            else
            {
                selectedCanvas.Children.Clear();
            }
        }

        private void deleteFileButton_Click(object sender, RoutedEventArgs e)
        {
            ListBoxItem item = (ListBoxItem)loadedFilesListBox.SelectedItem;
            if (item != null)
            {
                selectedCanvas.Children.Clear();
                LoadedImages.Remove((string)item.Content);
                loadedFilesListBox.Items.Remove(item);
            }
        }

    }
}
