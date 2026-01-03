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
            allToolStripMenuItem = new ToolStripMenuItem();
            replaceAllToolStripMenuItem = new ToolStripMenuItem();
            menuStrip1.SuspendLayout();
            SuspendLayout();
            // 
            // btnSearch
            // 
            btnSearch.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            btnSearch.Location = new Point(648, 50);
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
            txtResult.Location = new Point(2, 79);
            txtResult.Multiline = true;
            txtResult.Name = "txtResult";
            txtResult.ScrollBars = ScrollBars.Vertical;
            txtResult.Size = new Size(721, 188);
            txtResult.TabIndex = 1;
            // 
            // txtQuery
            // 
            txtQuery.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
            txtQuery.Location = new Point(2, 50);
            txtQuery.Name = "txtQuery";
            txtQuery.Size = new Size(640, 23);
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
            ragToolStripMenuItem.DropDownItems.AddRange(new ToolStripItem[] { allToolStripMenuItem, replaceAllToolStripMenuItem });
            ragToolStripMenuItem.Name = "ragToolStripMenuItem";
            ragToolStripMenuItem.Size = new Size(39, 20);
            ragToolStripMenuItem.Text = "Rag";
            // 
            // allToolStripMenuItem
            // 
            allToolStripMenuItem.Name = "allToolStripMenuItem";
            allToolStripMenuItem.Size = new Size(180, 22);
            allToolStripMenuItem.Text = "All";
            // 
            // replaceAllToolStripMenuItem
            // 
            replaceAllToolStripMenuItem.Name = "replaceAllToolStripMenuItem";
            replaceAllToolStripMenuItem.Size = new Size(180, 22);
            replaceAllToolStripMenuItem.Text = "Replace All";
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
            Text = "Form1";
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
        private ToolStripMenuItem allToolStripMenuItem;
        private ToolStripMenuItem replaceAllToolStripMenuItem;
    }
}
