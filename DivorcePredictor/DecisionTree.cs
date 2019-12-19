using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DivorcePredictor
{
    public class DecisionTree
    {
        public int m_classCount;
        public int m_attributeLen;
        public int m_maxDepth = 3;

        public DNode rootNode;
        public int Predict(int[] attributes)
        {
            var node = rootNode;
            while (node.leftNode != null)
            {
                var value = attributes[node.attributeIndex];
                if (value < node.threshold)
                {
                    node = node.leftNode;
                }
                else
                {
                    node = node.rightNode;
                }
            }

            return node.classIndex;
        }
        public void Train(int[][] X, int[] y)
        {
            var hs = new HashSet<int>();
            for (int i = 0; i < y.Length; i++)
                hs.Add(y[i]);

            m_classCount = hs.Count;

            m_attributeLen = X[0].Length;
            rootNode = InnerFit(X, y, 1);
            rootNode.isRoot = true;
        }



        private DNode InnerFit(int[][] X, int[] y , int depth )
        {
            // mevcut dugumun tuttugu samplelari siniflarina gore ayiriyoruz
            var samplesDistinct = new int[m_classCount];
            for(int i  =0; i < y.Length; i++)  samplesDistinct[y[i]]++;
            
            // maksimum sample sayisina sahip olan sinifi buluyoruz, (index sinifimizin ismi)
            var maxCountIndex = 0;
            for (int i = 1; i < m_classCount; i++)
                if (samplesDistinct[i] >= samplesDistinct[maxCountIndex])
                    maxCountIndex = i;



            var node = new DNode();

            node.gini = CalculateGI(samplesDistinct);
            node.samples = y.Length;
            node.attributeIndex = 0;
            node.threshold = 0;
            node.classIndex = maxCountIndex;

            if (depth <= m_maxDepth)
            {
                int splitIndex =  Split(X, y , out int fi , out float th);
                if(splitIndex > 0)
                {
                    node.attributeIndex = fi;
                    node.threshold = th;

                    var lX = X.Take(splitIndex).ToArray();
                    var ly = y.Take(splitIndex).ToArray();

                    var rX = X.Skip(splitIndex).ToArray();
                    var ry = y.Skip(splitIndex).ToArray();

                    node.leftNode =  InnerFit(lX, ly  , depth + 1);
                    node.leftNode.isLeft = true;
                    node.rightNode = InnerFit(rX, ry, depth + 1);
                    node.rightNode.isLeft = false;
                }
            }

            return node; 
        }
        private float CalculateGI(int[] sampleCounts)
        {
            int totalSampleCount = sampleCounts.Sum();
            float sum = 0;
            for (int i = 0; i < sampleCounts.Length; i++)
            {
                float g = (sampleCounts[i] / (float)totalSampleCount);
                sum += g * g;
            }
            return  1.0f - sum;
        }

        // agaci olabilecek en iyi yerden bolmek icin butun degerleri gini impurity ile olcuyoruz
        public int Split(int[][] X, int[] y , out int featureIndex, out float threshold)
        {
            if(y.Length > 1)
            {
                var sampleCount = y.Length;

                // mevcut samplelari siniflarina ayristirma olayi
                var samplesDistinct = new int[m_classCount];
                for (int i = 0; i < y.Length; i++) samplesDistinct[y[i]]++;

                // mevcut gini impurity
                float cur_gini = CalculateGI(samplesDistinct);

                // ayirgimiz index
                int splitNum = 0;
                // en iyi ozellik indexi
                int bestAttrIndex = 0;

                for(int attrIndex = 0; attrIndex < m_attributeLen; attrIndex++)
                {
                    // mevcut ozellikten ayiramayiz
                    Array.Sort(X, y, Comparer<int[]>.Create((a, b) => a[attrIndex] - b[attrIndex]));

                    var leftNodeSamples = new int[m_classCount];
                    
                    var rightNodeSamples = new int[m_classCount];
                    samplesDistinct.CopyTo(rightNodeSamples, 0);

                    for (int sampleNum = 1; sampleNum < sampleCount; sampleNum++ )
                    {
                    
                        var sample = y[sampleNum - 1];
                        leftNodeSamples[sample]++;
                        rightNodeSamples[sample]--;

                        // agaci esit degerlerden ayiramayiz...
                        if (X[sampleNum][attrIndex] != X[sampleNum - 1][attrIndex])
                        {
                            var gini_left = CalculateGI(leftNodeSamples);
                        var gini_right = CalculateGI(rightNodeSamples);

                        // 2 taraf icinde gini hesabi
                        float gini = (sampleNum * gini_left + (sampleCount - sampleNum) * gini_right) / sampleCount;

               
                        if (gini < cur_gini)
                        {
                            cur_gini = gini;
                            splitNum = sampleNum;
                            bestAttrIndex = attrIndex;
                        }

                        }
                    }
                }
                
                // split edilecek index bulunamadi
                if(splitNum == 0 || splitNum == sampleCount)
                {
                    featureIndex = bestAttrIndex;
                    threshold = 0;
                    return -1;
                }

                Array.Sort(X, y, Comparer<int[]>.Create((a, b) => a[bestAttrIndex] - b[bestAttrIndex]));
                featureIndex = bestAttrIndex;
                threshold = (X[splitNum][bestAttrIndex] + X[splitNum - 1][bestAttrIndex]) / (float)2;
                return splitNum;
            }
            else
            {
                featureIndex = 0;
                threshold = 0;
                return -1;
            }
        }


        public class DNode
        {
            public bool isLeft = false;
            public bool isRoot = false;

            public float gini;
            public float threshold;
            public int attributeIndex;
      

            public int classIndex;
            public int samples;
            public int instance_samples;

            public DNode leftNode;
            public DNode rightNode;

            public string Details { get
                {
                    string prefix = "";
                    if(!isRoot)
                        if (isLeft) prefix = "[LEFT]>";
                        else prefix = "[RIGHT]>";
                    
                    if (leftNode == null)
                        return prefix + string.Format("Leaf Samples:{0}, Class:{1}", samples, classIndex);
                    else
                        return prefix + string.Format("Node Samples:{0}, Attribute:{1}, Threshold:{2}, Class:{3}", samples, attributeIndex, threshold, classIndex);
                }
            }

            public IEnumerable<DNode> ChildNodes { get
                {
                    if (leftNode  != null) yield return leftNode;
                    if (rightNode != null) yield return rightNode;
                }
            }

 
        }

 



    }
}
