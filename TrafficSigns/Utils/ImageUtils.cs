using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Specific;

namespace TrafficSigns.Utils
{
    public class ImageUtils
    {
        public static readonly int ImageWidth = 32;
        public static readonly int ImageHeight = 32;

        //public static bool[,] GetImageArray(string filePath)
        //{
        //    var bitmap = new Bitmap(filePath);
        //    bool[,] result = new bool[bitmap.Width, bitmap.Height];

        //    for (int i = 0; i < bitmap.Width; i++)
        //    {
        //        for (int j = 0; j < bitmap.Height; j++)
        //        {
        //            Color pixelColor = bitmap.GetPixel(i, j);
        //            result[i, j] = (pixelColor.R > 127);
        //        }
        //    }

        //    return result;
        //}

        public static BiPolarMLData GetImageData(string filePath)
        {
            var bitmap = new Bitmap(filePath);
            var result = new BiPolarMLData(ImageWidth * ImageHeight);

            for (int i = 0; i < bitmap.Height; i++)
            {
                for (int j = 0; j < bitmap.Width; j++)
                {
                    Color pixelColor = bitmap.GetPixel(j, i);
                    bool value = (pixelColor.R > 127);
                    result.SetBoolean(i * bitmap.Width + j, value);
                }
            }

            return result;
        }

        //white noise distort
        public static void DistortImage(BiPolarMLData inputImage, double percentage)
        {
            Random randomGenerator = new Random();

            for (int i = 0; i < ImageWidth; i++)
            {
                for (int j = 0; j < ImageHeight; j++)
                {
                    bool orginal = inputImage.GetBoolean(i * ImageWidth + j);

                    if (randomGenerator.NextDouble() < percentage)
                    {
                        inputImage.SetBoolean(i * ImageWidth + j, !orginal);
                    }
                }
            }            
        }




    }
}
