using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Text;
using FastBertTokenizer;
using static AnthropicApp.Database.MemoryHashIndex;
using System.Diagnostics;

namespace AnthropicApp.Database
{
    public class EmbedderClassNew : IDisposable
    {
        private readonly BertTokenizer _tokenizer;
        private readonly InferenceSession _session;
        private readonly MemoryHashIndex _memoryIndex;
        private readonly RAGConfiguration _config;
        private readonly int _hiddenDim;
        private bool _disposed;

        private static readonly HashSet<string> StopWords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "it", "as", "that", "this", "these", "those", "are", "was", "were", "be"
        };

        public EmbedderClassNew(RAGConfiguration? config = null)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _tokenizer = new BertTokenizer();

            var sessionOptions = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                InterOpNumThreads = _config.InterOpNumThreads,
                IntraOpNumThreads = _config.IntraOpNumThreads
            };

            _session = new InferenceSession(_config.ModelPath, sessionOptions);

            // Determine hidden dimension from model metadata
            _hiddenDim = GetHiddenDimensionFromModel();

            using (var vocabReader = new StreamReader(_config.VocabularyPath))
            {
                _tokenizer.LoadVocabulary(vocabReader, convertInputToLowercase: true);
            }

            var hashOptions = new MemoryHashOptions
            {
                SimilarityThreshold = 0.5,
                MaxItems = 50000,
                HashSize = 32,
                NumHashFunctions = _config.NumberOfHashFunctions,
                NumBuckets = 256
            };
            _memoryIndex = new MemoryHashIndex(hashOptions);
        }

        private int GetHiddenDimensionFromModel()
        {
            // Try to get hidden dimension from output metadata
            // BERT large: 1024, BERT base: 768
            if (_session.OutputMetadata.TryGetValue("last_hidden_state", out var metadata))
            {
                var dims = metadata.Dimensions;
                if (dims != null && dims.Length >= 3)
                {
                    // Shape is [batch, seq_len, hidden_dim]
                    var hiddenDim = dims[2];
                    if (hiddenDim > 0)
                        return hiddenDim;
                }
            }

            // Fallback: assume BERT large (1024) or use config
            // Most ONNX BERT models use 768 (base) or 1024 (large)
            return _config.MaxSequenceLength; // Your config uses 512, adjust if needed
        }

        /// <summary>
        /// Generates embeddings for the given text using BERT.
        /// Returns the [CLS] token embedding (first token) which represents the entire sequence.
        /// </summary>
        public async Task<float[]> GetEmbeddingsAsync(string textToEmbed)
        {
            if (string.IsNullOrWhiteSpace(textToEmbed))
                throw new ArgumentException("Text to embed cannot be null or empty.", nameof(textToEmbed));

            return await GetEmbeddingsInternalAsync(textToEmbed);
        }

        /// <summary>
        /// Attempts to get embeddings, returning null on failure instead of throwing.
        /// </summary>
        public async Task<float[]?> TryGetEmbeddingsAsync(string textToEmbed)
        {
            if (string.IsNullOrWhiteSpace(textToEmbed))
                return null;

            try
            {
                return await GetEmbeddingsInternalAsync(textToEmbed);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Embedding generation failed: {ex.Message}");
                return null;
            }
        }

        private async Task<float[]> GetEmbeddingsInternalAsync(string textToEmbed)
        {
            var text = PreprocessText(RemoveStopWords(TrimWhitespace(textToEmbed)));

            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text is empty after preprocessing.", nameof(textToEmbed));

            var (inputIds, attentionMask, tokenTypeIds) = _tokenizer.Encode(text, _config.MaxSequenceLength, _config.MaxSequenceLength);

            var inputData = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids",
                    new DenseTensor<long>(inputIds, new[] { 1, _config.MaxSequenceLength })),
                NamedOnnxValue.CreateFromTensor("attention_mask",
                    new DenseTensor<long>(attentionMask, new[] { 1, _config.MaxSequenceLength }))
            };

            if (_session.InputMetadata.ContainsKey("token_type_ids"))
            {
                inputData.Add(NamedOnnxValue.CreateFromTensor("token_type_ids",
                    new DenseTensor<long>(tokenTypeIds, new[] { 1, _config.MaxSequenceLength })));
            }

            using var results = await Task.Run(() => _session.Run(inputData));

            var output = results.FirstOrDefault(r => r.Name == "last_hidden_state")
                      ?? results.FirstOrDefault();

            if (output == null)
                throw new InvalidOperationException("Model returned no output.");

            var tensor = output.AsTensor<float>();
            var embeddings = ExtractClsEmbedding(tensor);
            var normalizedEmbeddings = Normalize(embeddings);

            // Store in memory index for similarity search
            await _memoryIndex.AddOrUpdateAsync(
                id: Guid.NewGuid().ToString(),
                vector: normalizedEmbeddings,
                tags: new Dictionary<string, string> { { "text", textToEmbed } }
            );

            return normalizedEmbeddings;
        }

        /// <summary>
        /// Extracts the [CLS] token embedding from BERT's last_hidden_state output.
        /// BERT output shape: [batch_size, sequence_length, hidden_dim]
        /// The [CLS] token is at position 0 and represents the entire sequence.
        /// </summary>
        private float[] ExtractClsEmbedding(Tensor<float> tensor)
        {
            var dims = tensor.Dimensions.ToArray();

            if (dims.Length == 3)
            {
                // Shape: [batch=1, seq_len, hidden_dim]
                int hiddenDim = dims[2];
                var clsEmbedding = new float[hiddenDim];

                // Extract first token (CLS) embedding: tensor[0, 0, :]
                for (int i = 0; i < hiddenDim; i++)
                {
                    clsEmbedding[i] = tensor[0, 0, i];
                }

                return clsEmbedding;
            }
            else if (dims.Length == 2)
            {
                // Shape: [seq_len, hidden_dim] - single batch, take first row
                int hiddenDim = dims[1];
                var clsEmbedding = new float[hiddenDim];

                for (int i = 0; i < hiddenDim; i++)
                {
                    clsEmbedding[i] = tensor[0, i];
                }

                return clsEmbedding;
            }
            else if (dims.Length == 1)
            {
                // Already a 1D embedding (unlikely for BERT but handle it)
                return tensor.ToArray();
            }
            else
            {
                // Fallback: flatten and take first hidden_dim elements
                Debug.WriteLine($"Unexpected tensor shape: [{string.Join(", ", dims)}]. Using fallback extraction.");
                return tensor.ToArray().Take(_hiddenDim).ToArray();
            }
        }

        /// <summary>
        /// Splits text into overlapping chunks and generates embeddings for each.
        /// </summary>
        public async Task<List<float[]>> SplitStringIntoEmbeddingsListAsync(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be null or empty.", nameof(text));

            var chunks = SlidingWindow(text, _config.WordsPerString, _config.OverlapPercentage);
            var embeddingList = new List<float[]>(chunks.Count);

            foreach (var chunk in chunks)
            {
                try
                {
                    var embedding = await GetEmbeddingsAsync(chunk);
                    embeddingList.Add(embedding);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Failed to embed chunk: {ex.Message}");
                    // Continue with other chunks rather than failing entirely
                }
            }

            return embeddingList;
        }

        /// <summary>
        /// Searches the memory index for similar embeddings.
        /// </summary>
        public async Task<List<SearchResult>> SearchSimilarAsync(string text, int topK = 10)
        {
            var embeddings = await TryGetEmbeddingsAsync(text);
            if (embeddings == null)
                return new List<SearchResult>();

            return await _memoryIndex.SearchAsync(embeddings, topK);
        }

        /// <summary>
        /// Helper to generate and set a single embedding with timeout protection.
        /// </summary>
        public async Task HandleSingleEmbedding(string? content, bool embedFlag, Action<float[]?> setter)
        {
            if (string.IsNullOrEmpty(content) || !embedFlag)
                return;

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));

            try
            {
                var embedding = await GetEmbeddingsAsync(content);
                setter(embedding);
            }
            catch (OperationCanceledException)
            {
                Debug.WriteLine("Single embedding generation timed out.");
                setter(null);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Single embedding generation failed: {ex.Message}");
                setter(null);
            }
        }

        /// <summary>
        /// Helper to generate and set a list of embeddings with timeout protection.
        /// </summary>
        public async Task HandleListEmbedding(string? content, bool embedFlag, Action<List<float[]>?> setter)
        {
            if (string.IsNullOrEmpty(content) || !embedFlag)
                return;

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

            try
            {
                var embeddings = await SplitStringIntoEmbeddingsListAsync(content);
                setter(embeddings);
            }
            catch (OperationCanceledException)
            {
                Debug.WriteLine("List embedding generation timed out.");
                setter(null);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"List embedding generation failed: {ex.Message}");
                setter(null);
            }
        }

        #region Text Processing

        private List<string> SlidingWindow(string text, int wordsPerString, double overlapPercentage)
        {
            if (wordsPerString <= 0)
                throw new ArgumentException("Words per string must be greater than zero.", nameof(wordsPerString));

            if (overlapPercentage < 0 || overlapPercentage >= 100)
                throw new ArgumentOutOfRangeException(nameof(overlapPercentage), "Overlap percentage must be between 0 and 100.");

            var words = text.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);

            if (words.Length <= wordsPerString)
                return new List<string> { text };

            var result = new List<string>();
            int overlapWords = (int)Math.Floor(wordsPerString * (overlapPercentage / 100));
            int stepSize = Math.Max(1, wordsPerString - overlapWords);

            for (int i = 0; i < words.Length; i += stepSize)
            {
                int endIndex = Math.Min(i + wordsPerString, words.Length);
                result.Add(string.Join(" ", words.Skip(i).Take(endIndex - i)));

                if (endIndex >= words.Length)
                    break;
            }

            return result;
        }

        private string RemoveStopWords(string inputText)
        {
            if (string.IsNullOrWhiteSpace(inputText))
                return string.Empty;

            var words = inputText.Split(
                new[] { ' ', '.', ',', ';', ':', '!', '?', '\n', '\r', '\t' },
                StringSplitOptions.RemoveEmptyEntries);

            var filteredWords = words.Where(word => !StopWords.Contains(word));

            return string.Join(" ", filteredWords);
        }

        private static string TrimWhitespace(string input)
        {
            if (string.IsNullOrEmpty(input))
                return string.Empty;

            int startIndex = 0;
            int endIndex = input.Length - 1;

            // Skip leading whitespace
            while (startIndex <= endIndex && char.IsWhiteSpace(input[startIndex]))
                startIndex++;

            // Skip trailing whitespace (including \r and \n)
            while (endIndex >= startIndex && (char.IsWhiteSpace(input[endIndex])))
                endIndex--;

            if (startIndex > endIndex)
                return string.Empty;

            return input.Substring(startIndex, endIndex - startIndex + 1);
        }

        private static string PreprocessText(string input)
        {
            if (string.IsNullOrEmpty(input))
                return string.Empty;

            var normalized = input.Normalize(NormalizationForm.FormC);
            var sb = new StringBuilder(normalized.Length);

            foreach (char c in normalized)
            {
                if (!char.IsPunctuation(c))
                    sb.Append(c);
            }

            return sb.ToString();
        }

        #endregion

        #region Vector Operations

        private static float[] Normalize(float[] vector)
        {
            if (vector == null || vector.Length == 0)
                return Array.Empty<float>();

            double magnitude = 0;
            for (int i = 0; i < vector.Length; i++)
                magnitude += vector[i] * vector[i];

            magnitude = Math.Sqrt(magnitude);

            if (magnitude == 0)
                return new float[vector.Length]; // Return zero vector

            var result = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
                result[i] = (float)(vector[i] / magnitude);

            return result;
        }

        #endregion

        #region Disposal

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (disposing)
            {
                _session?.Dispose();
                _memoryIndex?.Dispose();
            }

            _disposed = true;
        }

        #endregion
    }
}