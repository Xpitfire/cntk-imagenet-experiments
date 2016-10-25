using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.MSR.CNTK.Extensibility.Managed;
using Microsoft.MSR.CNTK.Extensibility.Managed.CSEvalClient;

namespace CNTKDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            EvaluateImageClassificationModel(args.Length == 1 ? args[0] : null);
            Console.ReadLine();
        }

        /// <summary>
        /// This method shows how to evaluate a trained image classification model
        /// </summary>
        public static void EvaluateImageClassificationModel(string image = null)
        {
            try
            {
                // This example requires a pre-trained RestNet model.
                // The model can be downloaded from <see cref="https://www.cntk.ai/resnet/ResNet_152.model"/>
				// Or see the documentation page included other models <see cref="https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Classification/ResNet"/>
                string workingDirectory = Environment.CurrentDirectory;

                List<float> outputs;
                using (var model = new IEvaluateModelManagedF())
                {
                    // initialize model path
                    var modelFilePath = Path.Combine(workingDirectory, "models" , "ResNet_152.model");
                    // check if model is available
                    ThrowIfFileNotExist(modelFilePath,
                        $"Error: The model '{modelFilePath}' does not exist. Please download the ResNet model.");

                    // create network with the pre-trained model
                    model.CreateNetwork($"modelPath=\"{modelFilePath}\"", deviceId: -1);

                    // Prepare input value in the appropriate structure and size
                    var inDims = model.GetNodeDimensions(NodeGroup.Input);
                    if (inDims.First().Value != 224 * 224 * 3)
                    {
                        throw new CNTKRuntimeException(
                            $"The input dimension for {inDims.First()} is {inDims.First().Value} which is not the expected size of {224*224*3}.", string.Empty);
                    }

                    // initialize image either from command line or as hard-coded file
                    var imageFilePath = Path.Combine(workingDirectory, "images", image ?? "dog1-white_background.jpg");
                    ThrowIfFileNotExist(imageFilePath, $"Error: The test image file '{imageFilePath}' does not exist.");

                    // transform the image
                    var bmp = new Bitmap(Image.FromFile(imageFilePath));
                    var resized = bmp.Resize(224, 224, true);
                    var resizedCHW = resized.ParallelExtractCHW();
                    var inputs = new Dictionary<string, List<float>>
                    {
                        { inDims.First().Key, resizedCHW }
                    };

                    // call the evaluate method and get back the results (single layer output)...
                    var outDims = model.GetNodeDimensions(NodeGroup.Output);
                    outputs = model.Evaluate(inputs, outDims.First().Key);
                }
                
                // retrieve top 10 predictions
                var predictions = (from prediction in outputs.Select((value, index) => new { Value = value, Index = index })
                            orderby prediction.Value descending
                            select prediction).Take(10);

                // handle output via lookup table matching
                var lookupTablePath = Path.Combine(workingDirectory, "imagenet_words.txt");
                var lookupTable = File.ReadLines(lookupTablePath).ToList();
                var i = 0;
                foreach (var v in predictions)
                {
                    Console.WriteLine("Prediction {0} (Type: {1} | Ranked: {2}, File-Index: {3})", i++, lookupTable[v.Index].Substring(9), v.Value, v.Index);
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
            Console.WriteLine("Error: {0}\nNative CallStack: {1}\n Inner Exception: {2}", ex.Message, ex.NativeCallStack, ex.InnerException?.Message ?? "No Inner Exception");
            throw ex;
        }

        /// <summary>
        /// Handle general exceptions.
        /// </summary>
        /// <param name="ex">The exception to be handled.</param>
        private static void OnGeneralException(Exception ex)
        {
            // The pattern "Inner Exception" is used by End2EndTests to catch test failure.
            Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException?.Message ?? "No Inner Exception");
            throw ex;
        }

    }
}
