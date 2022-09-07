using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Diagnostics;

namespace ColorCounter
{
    internal class Program
    {
        static Action<Index1D, ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<long, Stride1D.Dense>>? countColorsKernel;
        static void Main(string[] args)
        {
            // Create GPU contexts
            Context context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());
            Accelerator device = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

            countColorsKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<long, Stride1D.Dense>>(CountColors);
            
            Console.WriteLine("Loading Image");
            Image<Rgb24> bitmap = Image.Load<Rgb24>("./test.jpeg");
            Console.WriteLine("Done Loading Image");

            TestGPU(device, bitmap);
            TestGPU(device, bitmap);
            TestCPU(bitmap);
            TestCPU(bitmap);

            Console.ReadLine();
        }

        static void TestGPU(Accelerator device, Image<Rgb24> bitmap)
        {
            Stopwatch loadTimer = Stopwatch.StartNew();
            
            using MemoryBuffer1D<long, Stride1D.Dense> output = device.Allocate1D<long>(2);
            using GPUImage image = new GPUImage(device, bitmap);

            loadTimer.Stop();

            Stopwatch timer = Stopwatch.StartNew();

            countColorsKernel!(image.bitmap.Width * image.bitmap.Height, image.data, output);
            device.Synchronize();
            long[] count = output.GetAsArray1D();
            
            timer.Stop();

            Console.WriteLine("Black Count: " + count[0] + " White Count: " + count[1]);
            Console.WriteLine("GPU load took: " + loadTimer.Elapsed.TotalMilliseconds + " ms");
            Console.WriteLine("GPU processing took: " + timer.Elapsed.TotalMilliseconds + " ms");
            Console.WriteLine("total GPU time: " + (timer.Elapsed + loadTimer.Elapsed).TotalMilliseconds + " ms");
        }

        static void TestCPU(Image<Rgb24> image)
        {
            Stopwatch timer = Stopwatch.StartNew();

            (long black, long white) = CountImageCPU(image);

            timer.Stop();

            Console.WriteLine("Black Count: " + black + " White Count: " + white);
            Console.WriteLine("CPU took: " + timer.Elapsed.TotalMilliseconds + " ms");
        }

        static (byte x, byte y, byte z) readFrameBuffer(ArrayView1D<byte, Stride1D.Dense> input, int index)
        {
            int subPixel = index * 3;
            return (input[subPixel], input[subPixel + 1], input[subPixel + 2]);
        }

        static void CountColors(Index1D pixel, ArrayView1D<byte, Stride1D.Dense> input, ArrayView1D<long, Stride1D.Dense> output)
        {
            (byte x, byte y, byte z) color = readFrameBuffer(input, pixel);

            // converting to gray scale on the GPU is likely faster
            // there are MANY ways to do it this one is based on the perceptual brightness of the subpixels
            int grayScale = (int)((color.x * 0.3f) + (color.y * 0.59f) + (color.z * 0.11f));
            
            // there are faster ways to do this but this will still be faster than on the CPU
            if (grayScale == 0) // black
            {
                Atomic.Add(ref output[0], 1);
            }
            else if(grayScale == 255) // white
            {
                Atomic.Add(ref output[1], 1);
            }
        }

        private static (byte x, byte y, byte z) readFrameBuffer(Rgb24[] input, int index)
        {
            return (input[index].R, input[index].G, input[index].B);
        }

        private static (long black, long white) CountImageCPU(Image<Rgb24> bitmap)
        {
            Rgb24[] rawPixelData = new Rgb24[bitmap.Height * bitmap.Width];
            Span<Rgb24> pixels = new Span<Rgb24>(rawPixelData);
            bitmap.CopyPixelDataTo(pixels);

            long blackCount = 0;
            long whiteCount = 0;

            Parallel.For(0, bitmap.Height, (int i) =>
            {
                for (int j = 0; j < bitmap.Width; j++)
                {
                    int pixel = i * bitmap.Width + j;
                    int subpixel = pixel * 3;

                    var color = readFrameBuffer(rawPixelData, pixel);

                    int grayScale = (int)((color.x * 0.3f) + (color.y * 0.59f) + (color.z * 0.11f));

                    // there are faster ways to do this but this will still be faster than on the CPU
                    if (grayScale == 0) // black
                    {
                        Interlocked.Increment(ref blackCount);
                    }
                    else if (grayScale == 255) // white
                    {
                        Interlocked.Increment(ref whiteCount);
                    }
                }
            });

            return (blackCount, whiteCount);
        }

        internal class GPUImage : IDisposable
        {
            public string path;
            public Image<Rgb24> bitmap;
            public byte[] pixelData;
            public MemoryBuffer1D<byte, Stride1D.Dense> data;

            public GPUImage(Accelerator device, string path)
            {
                this.path = path;
                bitmap = Image.Load<Rgb24>(path);
                CopyImageSlow();
                data = device.Allocate1D(pixelData);
            }

            public GPUImage(Accelerator device, Image<Rgb24> image)
            {
                this.path = "";
                bitmap = image;
                CopyImageSlow();
                data = device.Allocate1D(pixelData);
            }

            public void Dispose()
            {
                data.Dispose();
            }

            // There are faster ways to copy the Image if you copy the whole frame at
            // once, ImageSharp does not have as good as support as System.Drawing.Common
            private unsafe void CopyImageSlow()
            {
                Rgb24[] rawPixelData = new Rgb24[bitmap.Height * bitmap.Width];
                pixelData = new byte[bitmap.Height * bitmap.Width * 3];

                Span<Rgb24> pixels = new Span<Rgb24>(rawPixelData);
                bitmap.CopyPixelDataTo(pixels);

                Parallel.For(0, bitmap.Height, (int i) =>
                {
                    for (int j = 0; j < bitmap.Width; j++)
                    {
                        int pixel = i * bitmap.Width + j;
                        int subpixel = pixel * 3;

                        pixelData[subpixel] = rawPixelData[pixel].R;
                        pixelData[subpixel + 1] = rawPixelData[pixel].G;
                        pixelData[subpixel + 2] = rawPixelData[pixel].B;
                    }
                });
            }
        }
    }
}