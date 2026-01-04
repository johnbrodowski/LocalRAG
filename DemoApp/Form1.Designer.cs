namespace DemoApp
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            btnSearch = new Button();
            txtResult = new TextBox();
            txtQuery = new TextBox();
            btnRunTests = new Button();
            btnGenerateMockData = new Button();
            SuspendLayout();
            //
            // btnSearch
            //
            btnSearch.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            btnSearch.Location = new Point(616, 3);
            btnSearch.Name = "btnSearch";
            btnSearch.Size = new Size(75, 23);
            btnSearch.TabIndex = 0;
            btnSearch.Text = "Search";
            btnSearch.UseVisualStyleBackColor = true;
            btnSearch.Click += btnSearch_Click;
            //
            // txtResult
            //
            txtResult.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            txtResult.Location = new Point(2, 62);
            txtResult.Multiline = true;
            txtResult.Name = "txtResult";
            txtResult.ScrollBars = ScrollBars.Vertical;
            txtResult.Size = new Size(680, 150);
            txtResult.TabIndex = 1;
            //
            // txtQuery
            //
            txtQuery.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
            txtQuery.Location = new Point(2, 3);
            txtQuery.Name = "txtQuery";
            txtQuery.Size = new Size(599, 23);
            txtQuery.TabIndex = 2;
            txtQuery.Text = "Simple query that might have some matches in the rag database.";
            //
            // btnRunTests
            //
            btnRunTests.Location = new Point(2, 32);
            btnRunTests.Name = "btnRunTests";
            btnRunTests.Size = new Size(100, 23);
            btnRunTests.TabIndex = 3;
            btnRunTests.Text = "Run Tests";
            btnRunTests.UseVisualStyleBackColor = true;
            btnRunTests.Click += btnRunTests_Click;
            //
            // btnGenerateMockData
            //
            btnGenerateMockData.Location = new Point(108, 32);
            btnGenerateMockData.Name = "btnGenerateMockData";
            btnGenerateMockData.Size = new Size(120, 23);
            btnGenerateMockData.TabIndex = 4;
            btnGenerateMockData.Text = "Generate Mock Data";
            btnGenerateMockData.UseVisualStyleBackColor = true;
            btnGenerateMockData.Click += btnGenerateMockData_Click;
            //
            // Form1
            //
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(694, 224);
            Controls.Add(btnGenerateMockData);
            Controls.Add(btnRunTests);
            Controls.Add(txtQuery);
            Controls.Add(txtResult);
            Controls.Add(btnSearch);
            Name = "Form1";
            Text = "LocalRAG Test App";
            Load += Form1_Load;
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button btnSearch;
        private TextBox txtResult;
        private TextBox txtQuery;
        private Button btnRunTests;
        private Button btnGenerateMockData;
    }
}
