using System;
using System.Collections.Generic;
using System.IO;
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

namespace DivorcePredictor
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        public void GetData(string dataPath, out int[][] X, out int[] y, out string[] Xlabel, out string[] yLabel)
        {
            var lines = File.ReadAllLines(dataPath);

            var attribCount = lines[0].Split(';').Length - 1;
            X = new int[lines.Length - 1][];
            y = new int[lines.Length - 1];

            for (int i = 1; i < lines.Length; i++)
            {
                var line = lines[i];
                var values = line.Split(';');
                var attributes = new int[attribCount];
                for (int j = 0; j < values.Length - 1; j++)
                {
                    attributes[j] = int.Parse(values[j]);
                }
                X[i - 1] = attributes;
                y[i - 1] = int.Parse(values[values.Length - 1]);
            }

            string[] predicts =
            {
                "Yes","No"
            };
            string[] questions = {"If one of us apologizes when our discussion deteriorates, the discussion ends.",
                                  "I know we can ignore our differences, even if things get hard sometimes.",
                                  "When we need it, we can take our discussions with my spouse from the beginning and correct it.",
                                  "When I discuss with my spouse, to contact him will eventually work.",
                                  "The time I spent with my wife is special for us.",
                                  "We don't have time at home as partners.",
                                  "We are like two strangers who share the same environment at home rather than family.",
                                  "I enjoy our holidays with my wife.",
                                  "I enjoy traveling with my wife.",
                                  "Most of our goals are common to my spouse.",
                                  "I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.",
                                  "My spouse and I have similar values in terms of personal freedom.",
                                  "My spouse and I have similar sense of entertainment.",
                                  "Most of our goals for people (children, friends, etc.) are the same.",
                                  "Our dreams with my spouse are similar and harmonious.",
                                  "We're compatible with my spouse about what love should be.",
                                  "We share the same views about being happy in our life with my spouse",
                                  "My spouse and I have similar ideas about how marriage should be",
                                  "My spouse and I have similar ideas about how roles should be in marriage",
                                  "My spouse and I have similar values in trust.",
                                  "I know exactly what my wife likes.",
                                  "I know how my spouse wants to be taken care of when she/he sick.",
                                  "I know my spouse's favorite food.",
                                  "I can tell you what kind of stress my spouse is facing in her/his life.",
                                  "I have knowledge of my spouse's inner world.",
                                  "I know my spouse's basic anxieties.",
                                  "I know what my spouse's current sources of stress are.",
                                  "I know my spouse's hopes and wishes.",
                                  "I know my spouse very well.",
                                  "I know my spouse's friends and their social relationships.",
                                  "I feel aggressive when I argue with my spouse.",
                                  "When discussing with my spouse, I usually use expressions such as ‘you always’ or ‘you never’ .",
                                  "I can use negative statements about my spouse's personality during our discussions.",
                                  "I can use offensive expressions during our discussions.",
                                  "I can insult my spouse during our discussions.",
                                  "I can be humiliating when we discussions.",
                                  "My discussion with my spouse is not calm.",
                                  "I hate my spouse's way of open a subject.",
                                  "Our discussions often occur suddenly.",
                                  "We're just starting a discussion before I know what's going on.",
                                  "When I talk to my spouse about something, my calm suddenly breaks.",
                                  "When I argue with my spouse, ı only go out and I don't say a word.",
                                  "I mostly stay silent to calm the environment a little bit.",
                                  "Sometimes I think it's good for me to leave home for a while.",
                                  "I'd rather stay silent than discuss with my spouse.",
                                  "Even if I'm right in the discussion, I stay silent to hurt my spouse.",
                                  "When I discuss with my spouse, I stay silent because I am afraid of not being able to control my anger.",
                                  "I feel right in our discussions.",
                                  "I have nothing to do with what I've been accused of.",
                                  "I'm not actually the one who's guilty about what I'm accused of.",
                                  "I'm not the one who's wrong about problems at home.",
                                  "I wouldn't hesitate to tell my spouse about her/his inadequacy.",
                                  "When I discuss, I remind my spouse of her/his inadequacy.",
                                  "I'm not afraid to tell my spouse about her/his incompetence."};

            Xlabel = questions;
            yLabel = predicts;
        }

        public List<DecisionTree> KFoldTest(int[][] X, int[] y, int K = 10, bool alert = false)
        {
 


            int KSampleSize = (int)Math.Ceiling(y.Length /(float)K);


            List<DecisionTree> testTrees = new List<DecisionTree>();

           int correctPredicts = 0;
           int incorrectPredicts = 0;

            for (int i = 0; i < K; i++)
            {
                var X_validation = X.Skip(i * KSampleSize).Take(KSampleSize).ToArray();
                var y_validation = y.Skip(i * KSampleSize).Take(KSampleSize).ToArray();


                var x_remain = X.Skip((i + 1) * KSampleSize);
                var X_train = X.Take(i * KSampleSize).Concat(x_remain).ToArray();

                var y_remain = y.Skip((i + 1) * KSampleSize);
                var y_train = y.Take(i * KSampleSize).Concat(y_remain).ToArray();


                var testTree = new DecisionTree();
                testTree.Train(X_train, y_train);
                testTrees.Add(testTree);



                for (int j = 0; j < y_validation.Length; j++)
                {

                    if (testTree.Predict(X_validation[j]) == y_validation[j])
                        correctPredicts++;
                    else
                        incorrectPredicts++;
                }


            }
            var str = string.Format("Correct Predicts:{0} , IncorrectPredicts:{1} , avg acc:{2}, K-Num:{3}", correctPredicts, incorrectPredicts, correctPredicts / (float)(correctPredicts + incorrectPredicts) , K);
            Console.WriteLine(str);
            if(alert)
                MessageBox.Show(str , "Result");
            return testTrees;
        }

        int[][] X;
        int[] y;


        public MainWindow()
        {
            InitializeComponent();

            GetData("divorce.csv", out  X, out  y , out string[] Xlabel , out string[] ylabel);

            // veriseti randomizasyonu * kfold dan once
            var bytes = new byte[y.Length];
            new Random().NextBytes(bytes);
            var bytesC = (byte[])bytes.Clone();
            Array.Sort(bytes, X);
            Array.Sort(bytesC, y);

            KFoldTest(X , y ,20);
            KFoldTest(X, y,10);
            KFoldTest(X, y,5);
            KFoldTest(X, y,3);
            KFoldTest(X, y,2);


            tree = new DecisionTree();
            tree.Train(X, y);
 

            questionList = new List<Question>();
            for (int i = 0; i < Xlabel.Length; i++)
            {
                var q = new Question(i, Xlabel[i]);
                questionList.Add(q);
                xlist.Items.Add(q);
            }

            xtree.Items.Add(tree.rootNode);
 
 
        }

        List<Question> questionList;

        DecisionTree tree;
        private void predictButton_Click(object sender, RoutedEventArgs e)
        {
            var inputs = new int[questionList.Count];
            for (int i = 0; i < questionList.Count; i++)
            {
                inputs[i] = 3;

                if (questionList[i].A0) inputs[i] = 0;
                if (questionList[i].A1) inputs[i] = 1;
                if (questionList[i].A2) inputs[i] = 2;
                if (questionList[i].A3) inputs[i] = 3;
                if (questionList[i].A4) inputs[i] = 4;
            }


            var dnode = xtree.Items.GetItemAt(0) as DecisionTree.DNode;
            var dtree = new DecisionTree();
            dtree.rootNode = dnode;
            var pred = dtree.Predict(inputs);


            predictResultLabel.Foreground = Brushes.White;
            if (pred == 0)
            {
                predictResultLabel.Content = "Class:{0}, Label:Yes";
                
                predictResultLabel.Background = Brushes.Red;
            }
            else
            {
                predictResultLabel.Content = "Class:{1}, Label:No";

                predictResultLabel.Background = Brushes.Green;
            }




        }

        public class Question
        {
            public Question(int questionNum, string text)
            {
                Text = questionNum.ToString() + ". " + text;
                Group = questionNum.ToString();
            }

            public string Text { get; set; }
            public string Group { get; set; }

            public bool A0 { get; set; }
            public bool A1 { get; set; }
            public bool A2 { get; set; } = true;
            public bool A3 { get; set; }
            public bool A4 { get; set; }
        }


        List<DecisionTree> trees;
        private void Button_Click(object sender, RoutedEventArgs e)
        {
            var tag = int.Parse((KFolds.SelectedItem as ComboBoxItem).Tag.ToString());
            if(tag == 0)
            {
                treeComboBox.Items.Clear();

                var tree = new DecisionTree();
                tree.Train(X, y);

                trees = new List<DecisionTree>();
                trees.Add(tree);


                 var item = new ComboBoxItem();
                item.Content = "0";
                item.Tag = "0";

                treeComboBox.Items.Clear();
                treeComboBox.Items.Add(item);
                treeComboBox.SelectedIndex = 0;

            }
            else
            {
                int K = tag;

                trees = KFoldTest(X, y, K , true);
                treeComboBox.Items.Clear();

                for (int i = 0; i < K; i++)
                {
                    var item = new ComboBoxItem();
                    item.Content = i.ToString();
                    item.Tag = i.ToString();
                    treeComboBox.Items.Add(item);
                }
                treeComboBox.SelectedIndex = 0;
            }
            
        }

        private void Tree_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (treeComboBox.SelectedItem == null) return;

            int index = int.Parse((treeComboBox.SelectedItem as ComboBoxItem).Tag.ToString());

            xtree.Items.Clear();
            xtree.Items.Add(trees[index].rootNode);
        }
    }
}
