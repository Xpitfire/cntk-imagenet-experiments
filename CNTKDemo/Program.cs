using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.MSR.CNTK.Extensibility.Managed;
using Microsoft.MSR.CNTK.Extensibility.Managed.CSEvalClient;

namespace CNTKDemo
{
    class Program
    {
        static readonly Stopwatch StopWatch = new Stopwatch();

        const string CmdClassify = "-classify";
        const string CmdTransform = "-transform";
        static readonly string Error = $"Usage: CNTKDemo.exe ( {CmdClassify} | {CmdTransform} )";

        static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine(Error);
                VisualizeSourceCode();
                Console.ReadLine();
                return 1;
            }

            switch (args[0])
            {
                case CmdClassify:
                    EvaluateImageClassificationModel(args.Length == 2 ? args[1] : null);
                    break;
                case CmdTransform:
                    VisualizeSourceCode();
                    break;
                default:
                    Console.WriteLine(Error);
                    break;
            }
            
            Console.WriteLine("Finished execution!\nPress any key to exit...");
            Console.ReadLine();
            return 0;
        }
        
        /// <summary>
        /// This method shows how to evaluate a trained image classification model
        /// </summary>
        public static void EvaluateImageClassificationModel(string path = null)
        {
            try
            {
                // used to measure statistics
                TimeSpan initTime, transformTime, preditionTime;

                // This example requires a pre-trained RestNet model.
                // The model can be downloaded from <see cref="https://www.cntk.ai/resnet/ResNet_152.model"/>
                // Or see the documentation page included other models 
                // <see cref="https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Classification/ResNet"/>
                string workingDirectory = Environment.CurrentDirectory;

                List<float> outputs;
                using (var model = new IEvaluateModelManagedF())
                {
                    // initialize model path
                    var modelFilePath = Path.Combine(workingDirectory, "models", "ResNet_152.model");
                    // check if model is available
                    ThrowIfFileNotExist(modelFilePath,
                        $"Error: The model '{modelFilePath}' does not exist. Please download the ResNet model.");

                    StopWatch.Start();
                    // create network with the pre-trained model
                    model.CreateNetwork($"modelPath=\"{modelFilePath}\"", deviceId: -1);

                    // Prepare input value in the appropriate structure and size
                    var inDims = model.GetNodeDimensions(NodeGroup.Input);
                    if (inDims.First().Value != 224*224*3)
                    {
                        throw new CNTKRuntimeException(
                            $"The input dimension for {inDims.First()} is {inDims.First().Value} which is not the expected size of {224*224*3}.",
                            string.Empty);
                    }

                    StopWatch.Stop();
                    initTime = StopWatch.Elapsed;
                    StopWatch.Reset();
                    Console.WriteLine("\nInitialization time elapsed: {0}", initTime);

                    // initialize image either from command line or as hard-coded file
                    var images = Directory.GetFiles(path ?? Path.Combine(workingDirectory, "Visualizations"));
                    foreach (var img in images)
                    {
                        ThrowIfFileNotExist(img, $"Error: The test image file '{img}' does not exist.");

                        StopWatch.Start();
                        // Visualize Source
                        var imgData = new Bitmap(new MemoryStream(File.ReadAllBytes(img)));
                        var resizedCHW = imgData.ParallelExtractCHW();
                        StopWatch.Stop();
                        transformTime = StopWatch.Elapsed;
                        StopWatch.Reset();

                        var inputs = new Dictionary<string, List<float>>
                        {
                            { inDims.First().Key, resizedCHW }
                        };

                        StopWatch.Start();
                        // call the evaluate method and get back the results (single layer output)...
                        var outDims = model.GetNodeDimensions(NodeGroup.Output);
                        outputs = model.Evaluate(inputs, outDims.First().Key);
                        StopWatch.Stop();
                        preditionTime = StopWatch.Elapsed;
                        StopWatch.Reset();

                        // retrieve top 10 predictions
                        var predictions = (from prediction in outputs.Select((value, index) => new { Value = value, Index = index })
                                           orderby prediction.Value descending
                                           select prediction).Take(10);

                        // print report
                        Console.WriteLine($"\nREPORT: Classify image {img}");
                        Console.WriteLine("============================================");
                        Console.WriteLine("\n\nTransormation time elsapsed: {0}\nPrediciton time elapsed: {1}\n", transformTime, preditionTime);

                        // handle output via lookup table matching
                        var lookupTablePath = Path.Combine(workingDirectory, "imagenet_words.txt");
                        var lookupTable = File.ReadLines(lookupTablePath).ToList();
                        var i = 0;
                        foreach (var v in predictions)
                        {
                            Console.WriteLine("Prediction {0} (Type: {1} | Ranked: {2}, File-Index: {3})",
                                i++, lookupTable[v.Index].Substring(9), v.Value, v.Index);
                        }
                        Console.WriteLine("--------------------------------------------");
                        Console.WriteLine("\n");
                    }
                }
            }
            catch (CNTKException ex)
            {
                OnCNTKException(ex);
            }
            catch (Exception ex)
            {
                OnGeneralException(ex);
            }
        }

        private static void VisualizeSourceCode()
        {

            foreach (var i in Directory.GetFiles("Sources"))
            {
                // transform image
                var imgData = new ImageContainer
                {
                    Name = $"{Path.GetFileName(i)}.bmp",
                    RawData = Tokenize($"{i}")
                };
                StopWatch.Start();
                TransformToImage(imgData);
                RescaleImage(imgData);
                StopWatch.Stop();
                Console.WriteLine("File: {0}\nTransformation time elapsed: {1}\n", i, StopWatch.Elapsed);
                StopWatch.Reset();
                PersistImage(imgData);
            }
        }

        //public static Bitmap ToImage(byte[] data, string name)
        //{
        //    var bt = new Bitmap(data.Length, data.Length);
        //    for (int y = 0; y < bt.Height; y++)
        //    {
        //        for (int x = 0; x < bt.Width; x++)
        //        {
        //            var c = (data[(x + y) % data.Length] * data[(data.Length - x + y) % data.Length]) % 255;
        //            bt.SetPixel(y, x, Color.FromArgb(c, c, c));
        //        }
        //    }
        //    bt.Save(name);
        //    return bt;
        //}

        //public static Bitmap ToImage(int[] data, string name)
        //{
        //    var bt = new Bitmap(data.Length, data.Length);
        //    for (int y = 0; y < bt.Height; y++)
        //    {
        //        for (int x = 0; x < bt.Width; x++)
        //        {
        //            var r = data[(x + y) % data.Length] * data[(data.Length - x + y) % data.Length] % 256;
        //            var g = data[(x + y) % data.Length] * data[(x + y) % data.Length] % 256;
        //            var b = (int)Math.Abs(Math.Sin(r) * 256);
        //            bt.SetPixel(y, x, Color.FromArgb(r, g, b));
        //        }
        //    }
        //    bt.Save(name);
        //    return bt;
        //}

        public static void TransformToImage(ImageContainer imgData)
        {
            var dim = (int)Math.Sqrt(imgData.RawData.Length);
            var upscale = 65025;
            if ((float)imgData.RawData.Length/dim > 0) ++dim; 
            int min = imgData.RawData.Min();
            int max = imgData.RawData.Max() - min;

            var bt = new Bitmap(dim, dim);
            for (int y = 0; y < dim; y++)
            {
                for (int x = 0; x < dim; x++)
                {
                    int i = y*x + x;
                    if (i >= imgData.RawData.Length) continue;
                    
                    int d = (imgData.RawData[i] - min) * (upscale/max);
                    byte b1 = (byte)(d & 0xFF);
                    byte b2 = (byte)(d >> 8);
                    float r = b1;

                    float g = b2;
                    //float g = Sigmoid(d*d) * 255;
                    //var v = Math.Tanh(d);
                    //var vt = Math.Abs(v);
                    //float b = (float) (v < 0 ? vt*127 : vt*128 + 127);
                    //float b = (float) (Math.Abs(Math.Sin(d))*255); //Sigmoid(d / (float)max) * 255;
                    //float b = (float)(Math.Abs(Math.Sin(d)) * 255);

                    float b = (float)(Math.Abs(Math.Sin(b1 * b2))*255);
                    //Console.WriteLine("d: {0}, r: {1}, g: {2}, b: {3}", d, r, g, b);
                    bt.SetPixel(x, y, Color.FromArgb((int)r, (int)g, (int)b));
                }
            }
            imgData.Image = bt;
        }

        public static float Sigmoid(float x)
        {
            return 2 / (float)(1 + Math.Exp(-2 * x)) - 1;
        }

        public static int[] Tokenize(string fileName)
        {
            var rawData = File.ReadAllText(fileName);
            var tree = CSharpSyntaxTree.ParseText(rawData);
            var root = (CompilationUnitSyntax)tree.GetRoot();
            var tokens = root.DescendantTokens();
            return tokens.ToList().Select(x => x.RawKind).ToArray();
        }

        private static void PersistImage(ImageContainer imgData)
        {
            var subDir = "Visualizations";
            if (!Directory.Exists(subDir)) Directory.CreateDirectory(subDir);
            imgData.Image.Save(Path.Combine(Environment.CurrentDirectory, subDir, imgData.Name));
        }

        private static void RescaleImage(ImageContainer imgData)
        {
            imgData.Image = imgData.Image.Resize(224, 224, true);
        }

        public class ImageContainer
        {
            public string Name { get; set; }
            public int[] RawData { get; set; }
            public Bitmap Image { get; set; }
        }

        /// <summary>
        /// Checks whether the file exists. If not, write the error message on the console and throw FileNotFoundException.
        /// </summary>
        /// <param name="filePath">The file to check.</param>
        /// <param name="errorMsg">The message to write on console if the file does not exist.</param>
        private static void ThrowIfFileNotExist(string filePath, string errorMsg)
        {
            if (!File.Exists(filePath))
            {
                if (!string.IsNullOrEmpty(errorMsg))
                {
                    Console.WriteLine(errorMsg);
                }
                throw new FileNotFoundException($"File '{filePath}' not found.");
            }
        }

        /// <summary>
        /// Handle CNTK exceptions.
        /// </summary>
        /// <param name="ex">The exception to be handled.</param>
        private static void OnCNTKException(CNTKException ex)
        {
            // The pattern "Inner Exception" is used by End2EndTests to catch test failure.
            Console.WriteLine("Error: {0}\nNative CallStack: {1}\n Inner Exception: {2}",
                ex.Message, ex.NativeCallStack, ex.InnerException?.Message ?? "No Inner Exception");
            throw ex;
        }

        /// <summary>
        /// Handle general exceptions.
        /// </summary>
        /// <param name="ex">The exception to be handled.</param>
        private static void OnGeneralException(Exception ex)
        {
            // The pattern "Inner Exception" is used by End2EndTests to catch test failure.
            Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}",
                ex.Message, ex.StackTrace, ex.InnerException?.Message ?? "No Inner Exception");
            throw ex;
        }

    }
}
