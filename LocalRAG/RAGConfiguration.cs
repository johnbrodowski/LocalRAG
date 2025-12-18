using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnthropicApp.Database
{
    public class RAGConfiguration
    {
        public string DatabasePath { get; set; } = "Database\\Memory\\FeedbackEmbeddings512.db";
        public string ModelPath { get; set; } = "onnxBERT\\model2.onnx";
        public string VocabularyPath { get; set; } = "Vocabularies\\base_uncased_large.txt";
        public int MaxSequenceLength { get; set; } = 512;
        public int WordsPerString { get; set; } = 40;
        public double OverlapPercentage { get; set; } = 15;
        public int NumberOfHashFunctions { get; set; } = 8;
        public int NumberOfHashTables { get; set; } = 10;
        public int MaxQueueSize { get; set; } = 1000;
        public int MaxRetryAttempts { get; set; } = 3;
        public int RetryDelayMs { get; set; } = 1000;
        public int InterOpNumThreads { get; set; } = 32;
        public int IntraOpNumThreads { get; set; } = 2;
 
        public int MaxCacheItems { get; set; } = 10000;
        public long CacheItemSizeThreshold { get; set; } = 1024 * 1024; // 1MB
        public TimeSpan CacheExpiry { get; set; } = TimeSpan.FromMinutes(15);
 
    }

}
