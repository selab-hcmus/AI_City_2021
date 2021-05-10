using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp31
{
    class Program
    {
        static void Main(string[] args)
        {
            string[] filenames = System.IO.Directory.GetFiles(@"c:\Temp\Box\test_track_boxes_ver2", "*.txt");
            Console.WriteLine("Value for k (default: 5)");
            k = int.Parse(Console.ReadLine());
            Console.WriteLine("Value for delta (default: 5)");
            delta = int.Parse(Console.ReadLine());
            Console.WriteLine("Value for alpha (default: 0.1)");
            alpha = double.Parse(Console.ReadLine());

            for (int i=0; i<filenames.Length; i++)
            {
                ProcessItem(filenames[i]);
            }
        }

        private static void ProcessItem(string strFilename)
        {
            StreamReader sr = new StreamReader(strFilename);
            int n = int.Parse(sr.ReadLine()); //n: number of frames
            sr.ReadLine();
            int a, b, c, d;
            int[] x = new int[n];
            int[] y = new int[n];
            int mx = 0, my = 0;
            for (int i=0; i<n; i++)
            {
                string s;
                s = sr.ReadLine();
                string[] ss = s.Split(' ');
                a = int.Parse(ss[0]);
                b = int.Parse(ss[1]);
                c = int.Parse(ss[2]);
                d = int.Parse(ss[3]);
                x[i] = (a + c) / 2;
                y[i] = (b + d) / 2;
                if (x[i] > mx) mx = x[i];
                if (y[i] > my) my = y[i];

            }

            if (Evaluate(strFilename, n, x, y))
                Console.WriteLine(strFilename);

            //RenderResult(strFilename, n, x, y, mx, my);
            sr.Close();
        }
        private static int k = 5;
        private static double alpha = 0.1;
        private static bool Evaluate(string strFilename, int n, int[] x, int[] y)
        {
            if (n <= k)
                return false;
            double[] d = new double[n - k]; //k: skip frames
            for (int i=0; i<n-k; i++)
            {
                d[i] = Distance(x[i], y[i], x[i + k], y[i + k]); // Euclid distance
            }

            d = Smooth(d);

            double meand = CalculateMean(d);

            for (int i = 0; i < n - k; i++)
                if (d[i] < meand * alpha)
                {
                    Console.WriteLine(i.ToString());
                    return true;
                }
            return false;
        }

        static int delta = 5;
        private static double[] Smooth(double[] d)
        {
            
            int n = d.Length;
            if (n <= delta)
                return d;
            double[] res = new double[n];

            double s;
            for (int i=0; i<n-delta; i++)
            {
                s = 0;
                for (int j = 0; j < delta; j++)
                    s += d[i + j];
                res[i] = s / delta;
            }
            for (int i = n - delta; i < n; i++)
                res[i] = res[n - delta - 1];
            return res;
        }

        private static double CalculateMean(double[] d)
        {
            int n = d.Length;
            double x = 0;
            for (int i = 0; i < n; i++)
                x += d[i];
            x /= n;
            return x;
        }

        private static double Distance(int v1, int v2, int v3, int v4)
        {
            return Math.Sqrt((v3 - v1) * (v3 - v1) + (v4 - v2) * (v4 - v2));
        }

        private static void RenderResult(string strFilename, int n, int[] x, int[] y, int mx, int my)
        {
            Bitmap bmp = new Bitmap(mx+100, my+100);
            Graphics g = Graphics.FromImage(bmp);
            g.Clear(Color.White);

            g.FillRectangle(br, new Rectangle(new Point(x[0], y[0]), new Size(20, 20)));
            for (int i=0; i<n-1; i++)
            {
                DrawLine(g,
                         x[i],
                         y[i],
                         x[i + 1],
                         y[i + 1]);
            }
            bmp.Save(strFilename + ".bmp");
        }
        static Brush br = new SolidBrush(Color.Blue);

        static Pen p = new Pen(Color.Red);

        private static void DrawLine(Graphics g, int v1, int v2, int v3, int v4)
        {
            g.DrawLine(p, v1, v2, v3, v4);
        }
    }
}
