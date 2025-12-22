using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalRAG
{
    /// <summary>
    /// Configuration settings for the LocalRAG system.
    /// </summary>
    public class RAGConfiguration
    {
        private static string GetAppPath(string relativePath)
        {
            var basePath = AppDomain.CurrentDomain.BaseDirectory;
            return Path.Combine(basePath, relativePath);
        }

        /// <summary>
        /// Path to the SQLite database file for storing embeddings. Will be created if it doesn't exist.
        /// </summary>
        public string DatabasePath { get; set; } = GetAppPath(Path.Combine("Database", "Memory", "FeedbackEmbeddings512.db"));

        /// <summary>
        /// Path to the ONNX BERT model file. Download a BERT model in ONNX format and update this path.
        /// Example: https://huggingface.co/models?library=onnx&search=bert
        /// </summary>
        public string ModelPath { get; set; } = GetAppPath(Path.Combine("onnxBERT", "model2.onnx"));

        /// <summary>
        /// Path to the BERT vocabulary file (vocab.txt). Should match the BERT model being used.
        /// </summary>
        public string VocabularyPath { get; set; } = GetAppPath(Path.Combine("Vocabularies", "base_uncased_large.txt"));
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
