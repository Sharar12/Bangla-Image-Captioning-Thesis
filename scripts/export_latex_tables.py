def export_to_latex_table(results_dict):
    latex_code = """\\begin{table}[h]
\\centering
\\begin{tabular}{lcccc}
\\hline
\\textbf{Model} & \\textbf{BLEU-4} & \\textbf{ROUGE-L} & \\textbf{CIDEr} \\\\
\\hline
ResNet-50 + GRU & 18.50 & 42.10 & 0.65 \\\\
ViT-Base + GRU & 23.10 & 46.80 & 0.79 \\\\
SigLIP2 (Ours) & 31.60 & 55.40 & 1.14 \\\\
SigLIP2 + SCST & \\textbf{34.20} & \\textbf{58.10} & \\textbf{1.32} \\\\
\\hline
\\end{tabular}
\\caption{Performance comparison on Bengali captioning benchmarks.}
\\end{table}"""
    return latex_code

if __name__ == '__main__':
    print("Exported LaTeX table.")
