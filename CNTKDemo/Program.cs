using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
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
                return 1;
            }

            switch (args[0])
            {
                // classify images
                case CmdClassify:
                    EvaluateImageClassificationModel(args.Length == 2 ? args[1] : null);
                    break;
                // transform source to images
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

        /// <summary>
        /// Transform source files to bitmap.
        /// </summary>
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
        
        /// <summary>
        /// Calculate transformation of the raw data representation of a ImageContainer from source
        /// code tokens to a bitmap representation.
        /// </summary>
        /// <param name="imgData"></param>
        public static void TransformToImage(ImageContainer imgData)
        {
            // Codes: http://source.roslyn.io/#Microsoft.CodeAnalysis.CSharp/Syntax/SyntaxKind.cs,7c4040782a1b2ce0
            const int normalizationMinValue = 8000; // Roslyn token min value
            const int normalizationMaxValue = 9052; // Roslyn token max value

            const int upscale = 65025; // upscal resolution

            // calculate image dimension
            var dim = (int)Math.Sqrt(imgData.RawData.Length);
            // check if odd or even number and add extra dimension if odd
            if ((float)imgData.RawData.Length/dim > 0) ++dim;
            // compute bounds
            int min = normalizationMinValue;
            int max = normalizationMaxValue - min;

            // initialize empty image
            var bt = new Bitmap(dim, dim);
            for (int y = 0; y < dim; y++) // y-axis
            {
                for (int x = 0; x < dim; x++) // x-axis
                {
                    int i = y*x + x; // compute vector index
                    // skip over-dimension position if max vector-index reached
                    if (i >= imgData.RawData.Length) continue;
                    
                    // re-scale from token range values to RGB specturm
                    int d = (imgData.RawData[i] - min) * (upscale/max);
                    // compute colors
                    float r = (byte)(d >> 8); // use higher bits
                    float g = (byte)(d & 0xFF); // use lower 8 bits
                    float b = (float)(Math.Abs(Math.Sin(r * g))*255);
                    // set new features
                    bt.SetPixel(x, y, Color.FromArgb((int)r, (int)g, (int)b));
                }
            }
            imgData.Image = bt;
        }
        
        /// <summary>
        /// Translate from source file to Roslyn tokens int-values.
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public static int[] Tokenize(string fileName)
        {
            var rawData = File.ReadAllText(fileName);
            var tree = CSharpSyntaxTree.ParseText(rawData);
            var root = (CompilationUnitSyntax)tree.GetRoot();
            var tokens = root.DescendantTokens();
            return tokens.ToList().Select(x => x.RawKind).ToArray();
        }

        /// <summary>
        /// Save image container bitmap.
        /// </summary>
        /// <param name="imgData"></param>
        private static void PersistImage(ImageContainer imgData)
        {
            var subDir = "Visualizations";
            if (!Directory.Exists(subDir)) Directory.CreateDirectory(subDir);
            imgData.Image.Save(Path.Combine(Environment.CurrentDirectory, subDir, imgData.Name));
        }

        /// <summary>
        /// Rescale image to normalized size.
        /// </summary>
        /// <param name="imgData"></param>
        private static void RescaleImage(ImageContainer imgData)
        {
            imgData.Image = imgData.Image.Resize(224, 224, true);
        }

        /// <summary>
        /// Data container class.
        /// </summary>
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
