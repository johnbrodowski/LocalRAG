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
            menuStrip1 = new MenuStrip();
            fileToolStripMenuItem = new ToolStripMenuItem();
            ragToolStripMenuItem = new ToolStripMenuItem();
            backfillToolStripMenuItem = new ToolStripMenuItem();
            missingToolStripMenuItem = new ToolStripMenuItem();
            allToolStripMenuItem = new ToolStripMenuItem();
            menuStrip1.SuspendLayout();
            SuspendLayout();
            // 
            // btnSearch
            // 
            btnSearch.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            btnSearch.Location = new Point(659, 26);
            btnSearch.Name = "btnSearch";
            btnSearch.Size = new Size(75, 26);
            btnSearch.TabIndex = 0;
            btnSearch.Text = "Search";
            btnSearch.UseVisualStyleBackColor = true;
            btnSearch.Click += btnSearch_Click;
            // 
            // txtResult
            // 
            txtResult.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            txtResult.Location = new Point(5, 56);
            txtResult.Multiline = true;
            txtResult.Name = "txtResult";
            txtResult.ScrollBars = ScrollBars.Vertical;
            txtResult.Size = new Size(726, 219);
            txtResult.TabIndex = 1;
            // 
            // txtQuery
            // 
            txtQuery.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
            txtQuery.Font = new Font("Segoe UI", 10F);
            txtQuery.Location = new Point(6, 28);
            txtQuery.Name = "txtQuery";
            txtQuery.Size = new Size(650, 25);
            txtQuery.TabIndex = 2;
            txtQuery.Text = "Simple query that might have some matches in the rag database.";
            // 
            // menuStrip1
            // 
            menuStrip1.Items.AddRange(new ToolStripItem[] { fileToolStripMenuItem, ragToolStripMenuItem });
            menuStrip1.Location = new Point(0, 0);
            menuStrip1.Name = "menuStrip1";
            menuStrip1.Size = new Size(735, 24);
            menuStrip1.TabIndex = 3;
            menuStrip1.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            fileToolStripMenuItem.Size = new Size(37, 20);
            fileToolStripMenuItem.Text = "File";
            // 
            // ragToolStripMenuItem
            // 
            ragToolStripMenuItem.DropDownItems.AddRange(new ToolStripItem[] { backfillToolStripMenuItem });
            ragToolStripMenuItem.Name = "ragToolStripMenuItem";
            ragToolStripMenuItem.Size = new Size(39, 20);
            ragToolStripMenuItem.Text = "Rag";
            // 
            // backfillToolStripMenuItem
            // 
            backfillToolStripMenuItem.DropDownItems.AddRange(new ToolStripItem[] { missingToolStripMenuItem, allToolStripMenuItem });
            backfillToolStripMenuItem.Name = "backfillToolStripMenuItem";
            backfillToolStripMenuItem.Size = new Size(180, 22);
            backfillToolStripMenuItem.Text = "Backfill";
            // 
            // missingToolStripMenuItem
            // 
            missingToolStripMenuItem.Name = "missingToolStripMenuItem";
            missingToolStripMenuItem.Size = new Size(180, 22);
            missingToolStripMenuItem.Text = "Missing";
            // 
            // allToolStripMenuItem
            // 
            allToolStripMenuItem.Name = "allToolStripMenuItem";
            allToolStripMenuItem.Size = new Size(180, 22);
            allToolStripMenuItem.Text = "All";
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(735, 279);
            Controls.Add(txtQuery);
            Controls.Add(txtResult);
            Controls.Add(btnSearch);
            Controls.Add(menuStrip1);
            MainMenuStrip = menuStrip1;
            Name = "Form1";
            Text = "LocalRAG Test App";
            Load += Form1_Load;
            menuStrip1.ResumeLayout(false);
            menuStrip1.PerformLayout();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button btnSearch;
        private TextBox txtResult;
        private TextBox txtQuery;
        private MenuStrip menuStrip1;
        private ToolStripMenuItem fileToolStripMenuItem;
        private ToolStripMenuItem ragToolStripMenuItem;
        private ToolStripMenuItem backfillToolStripMenuItem;
        private ToolStripMenuItem missingToolStripMenuItem;
        private ToolStripMenuItem allToolStripMenuItem;
    }
}
