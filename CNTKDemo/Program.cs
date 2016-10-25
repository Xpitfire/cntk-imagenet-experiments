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
            EvaluateImageClassificationModel();
            Console.ReadLine();
        }

        /// <summary>
        /// This method shows how to evaluate a trained image classification model
        /// </summary>
        public static void EvaluateImageClassificationModel()
        {
            try
            {
                // This example requires the RestNet_18 model.
                // The model can be downloaded from <see cref="https://www.cntk.ai/resnet/ResNet_18.model"/>
                // The model is assumed to be located at: <CNTK>\Examples\Image\Classification\ResNet 
                // along with a sample image file named "zebra.jpg".
                //string workingDirectory = Path.Combine(Environment.CurrentDirectory, @"..\..\Examples\Image\Classification\ResNet");
                string workingDirectory = Environment.CurrentDirectory;

                List<float> outputs;

                using (var model = new IEvaluateModelManagedF())
                {
                    string modelFilePath = Path.Combine(workingDirectory, "ResNet_152.model");
                    ThrowIfFileNotExist(modelFilePath,
                        string.Format("Error: The model '{0}' does not exist. Please download the model from https://www.cntk.ai/resnet/ResNet_18.model and save it under ..\\..\\Examples\\Image\\Classification\\ResNet.", modelFilePath));

                    model.CreateNetwork(string.Format("modelPath=\"{0}\"", modelFilePath), deviceId: -1);

                    // Prepare input value in the appropriate structure and size
                    var inDims = model.GetNodeDimensions(NodeGroup.Input);
                    if (inDims.First().Value != 224 * 224 * 3)
                    {
                        throw new CNTKRuntimeException(string.Format("The input dimension for {0} is {1} which is not the expected size of {2}.", inDims.First(), inDims.First().Value, 224 * 224 * 3), string.Empty);
                    }

                    // Transform the image
                    string imageFileName = Path.Combine(workingDirectory, "dog1-white_background.jpg");
                    ThrowIfFileNotExist(imageFileName, string.Format("Error: The test image file '{0}' does not exist.", imageFileName));

                    Bitmap bmp = new Bitmap(Bitmap.FromFile(imageFileName));

                    var resized = bmp.Resize(224, 224, true);
                    var resizedCHW = resized.ParallelExtractCHW();
                    var inputs = new Dictionary<string, List<float>>() { { inDims.First().Key, resizedCHW } };

                    // We can call the evaluate method and get back the results (single layer output)...
                    var outDims = model.GetNodeDimensions(NodeGroup.Output);
                    outputs = model.Evaluate(inputs, outDims.First().Key);
                }

                // Retrieve the outcome index (so we can compare it with the expected index)
                var max = outputs.Select((value, index) => new { Value = value, Index = index })
                    .Aggregate((a, b) => (a.Value > b.Value) ? a : b)
                    .Index;
                
                var predictions = (from prediction in outputs.Select((value, index) => new { Value = value, Index = index })
                            orderby prediction.Value descending
                            select prediction).Take(10);

                // Lookup table link: https://github.com/sh1r0/caffe-android-demo/blob/master/app/src/main/assets/synset_words.txt#L1
                foreach (var v in predictions)
                {
                    Console.WriteLine("Prediction: Index: {0} Value: {1}", v.Index, v.Value);
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
                throw new FileNotFoundException(string.Format("File '{0}' not found.", filePath));
            }
        }

        /// <summary>
        /// Handle CNTK exceptions.
        /// </summary>
        /// <param name="ex">The exception to be handled.</param>
        private static void OnCNTKException(CNTKException ex)
        {
            // The pattern "Inner Exception" is used by End2EndTests to catch test failure.
            Console.WriteLine("Error: {0}\nNative CallStack: {1}\n Inner Exception: {2}", ex.Message, ex.NativeCallStack, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            throw ex;
        }

        /// <summary>
        /// Handle general exceptions.
        /// </summary>
        /// <param name="ex">The exception to be handled.</param>
        private static void OnGeneralException(Exception ex)
        {
            // The pattern "Inner Exception" is used by End2EndTests to catch test failure.
            Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            throw ex;
        }

    }
}
