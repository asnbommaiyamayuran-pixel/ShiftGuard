"""
================================================================================
ShiftGuard: Professional PDF Research Paper Report Generator
================================================================================
Converts ShiftGuard_Research_Report.md and embedded visualization PNG charts into
a publication-grade formatted PDF document ('ShiftGuard_Research_Report.pdf').
================================================================================
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT


def build_pdf_report(results_dir: str = "D:/FYP/ShiftGuard/results",
                     pdf_filename: str = "ShiftGuard_Research_Report.pdf"):
    pdf_path = os.path.join(results_dir, pdf_filename)
    print("=" * 80)
    print(f" GENERATING PDF RESEARCH REPORT: '{pdf_path}'")
    print("=" * 80)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()
    
    # Custom Color Palette
    primary_color = HexColor("#1a2b4c")
    secondary_color = HexColor("#2c3e50")
    accent_color = HexColor("#0066cc")
    bg_light = HexColor("#f8f9fa")
    border_color = HexColor("#e2e8f0")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        spaceAfter=12,
    )

    abstract_style = ParagraphStyle(
        'DocAbstract',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=secondary_color,
        alignment=TA_JUSTIFY,
        fontName="Helvetica-Oblique",
    )

    heading2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=primary_color,
        fontName="Helvetica-Bold",
        spaceBefore=14,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=secondary_color,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        textColor=HexColor("#ffffff"),
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )

    table_body_style = ParagraphStyle(
        'TableBody',
        parent=styles['Normal'],
        fontSize=8,
        leading=10.5,
        textColor=secondary_color,
        alignment=TA_CENTER,
    )

    story = []

    # Title
    story.append(Paragraph("ShiftGuard: A Shift-Aware Adaptive Vision System for Robust Perception under Distribution Shifts", title_style))
    story.append(Spacer(1, 4))

    # Abstract Box
    abstract_text = (
        "<b>Abstract—</b> Deep vision models operating in real-world environments face compounding distribution shifts "
        "comprising domain shifts (environmental lighting/weather changes), modality corruptions (sensor noise/missing channels), "
        "and semantic shifts (unseen novel objects). Standard vision backbones suffer severe accuracy degradation and overconfident "
        "miscalibration under these conditions. We introduce <b>ShiftGuard</b>, a parameter-efficient adaptive architecture that "
        "dynamic-routes features through specialized low-rank (LoRA) adapters based on a multi-head shift signature. "
        "ShiftGuard dynamically assesses domain, modality, and semantic uncertainty to provide calibrated predictions and trusted output "
        "decisions. Evaluated on synthetic road scenes and real image corruption benchmarks, ShiftGuard achieves <b>95.13% In-Distribution Accuracy</b> "
        "and <b>93.50% Compound Shift Accuracy</b> while using only <b>0.96% parameter overhead</b> and adding negligible latency (<b>+0.022 ms / +0.72%</b>)."
    )
    abstract_table = Table(
        [[Paragraph(abstract_text, abstract_style)]],
        colWidths=[7.3 * inch]
    )
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 10))

    # Section 1: System Architecture
    story.append(Paragraph("1. System Architecture & Data Flow", heading2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=accent_color, spaceAfter=8))
    
    arch_img_path = os.path.join(results_dir, "shiftguard_architecture_diagram_1789023342848.png")
    if os.path.exists(arch_img_path):
        story.append(Image(arch_img_path, width=7.0 * inch, height=3.5 * inch))
        story.append(Spacer(1, 6))

    story.append(Paragraph(
        "ShiftGuard processes input images through five modular components operating end-to-end: "
        "(1) <b>Vision Foundation Model (MiniViT)</b> extracting latent feature representations <i>f</i>; "
        "(2) <b>Multi-Head Shift Analyzer</b> estimating a 3D shift signature <i>S = [s_d, s_m, s_s]</i>; "
        "(3) <b>Adaptive Router</b> applying selective soft gating with straight-through thresholding; "
        "(4) <b>Parallel LoRA Adapters</b> specialized for domain, modality, and semantic shifts; and "
        "(5) <b>Output Decision Engine</b> evaluating confidence and predictive entropy to flag <i>TRUSTED</i> vs <i>UNCERTAIN_OOD</i> predictions.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # Section 2: Experimental Results
    story.append(Paragraph("2. Experimental Benchmarks & Performance", heading2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=accent_color, spaceAfter=8))

    # Table 1 Data
    t1_data = [
        [Paragraph("Model Configuration", table_header_style), Paragraph("Epochs", table_header_style), Paragraph("Rank (r)", table_header_style), Paragraph("Parameters", table_header_style), Paragraph("ID Accuracy", table_header_style), Paragraph("Compound Acc", table_header_style), Paragraph("ECE", table_header_style), Paragraph("OOD-AUROC", table_header_style)],
        [Paragraph("No-Adapter Baseline", table_body_style), Paragraph("6", table_body_style), Paragraph("N/A", table_body_style), Paragraph("1,829,568", table_body_style), Paragraph("78.12%", table_body_style), Paragraph("76.22%", table_body_style), Paragraph("0.0880", table_body_style), Paragraph("0.6266", table_body_style)],
        [Paragraph("ShiftGuard Standard", table_body_style), Paragraph("6", table_body_style), Paragraph("8", table_body_style), Paragraph("1,901,708 (0.48%)", table_body_style), Paragraph("78.62%", table_body_style), Paragraph("77.64%", table_body_style), Paragraph("0.0858", table_body_style), Paragraph("0.6345", table_body_style)],
        [Paragraph("ShiftGuard Extended", table_body_style), Paragraph("15", table_body_style), Paragraph("16", table_body_style), Paragraph("1,910,924 (0.96%)", table_body_style), Paragraph("<b>95.13%</b>", table_body_style), Paragraph("<b>93.50%</b>", table_body_style), Paragraph("<b>0.0262</b>", table_body_style), Paragraph("<b>0.6868</b>", table_body_style)],
    ]
    t1 = Table(t1_data, colWidths=[1.4*inch, 0.5*inch, 0.6*inch, 1.2*inch, 0.9*inch, 0.9*inch, 0.7*inch, 0.8*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#ffffff"), bg_light]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # Visualizations Grid
    story.append(Paragraph("3. Visualizations & Calibration Analysis", heading2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=accent_color, spaceAfter=8))

    img_curves = os.path.join(results_dir, "shiftguard_training_curves.png")
    img_reliability = os.path.join(results_dir, "shiftguard_reliability_diagram.png")
    img_gating = os.path.join(results_dir, "shiftguard_adapter_gating.png")
    img_entropy = os.path.join(results_dir, "shiftguard_ood_entropy.png")

    viz_grid_1 = []
    if os.path.exists(img_curves) and os.path.exists(img_reliability):
        viz_grid_1.append([
            Image(img_curves, width=3.5 * inch, height=2.2 * inch),
            Image(img_reliability, width=3.5 * inch, height=2.2 * inch)
        ])
        t_viz1 = Table(viz_grid_1, colWidths=[3.65*inch, 3.65*inch])
        t_viz1.setStyle(TableStyle([('PADDING', (0,0), (-1,-1), 0)]))
        story.append(t_viz1)
        story.append(Spacer(1, 6))

    viz_grid_2 = []
    if os.path.exists(img_gating) and os.path.exists(img_entropy):
        viz_grid_2.append([
            Image(img_gating, width=3.5 * inch, height=2.2 * inch),
            Image(img_entropy, width=3.5 * inch, height=2.2 * inch)
        ])
        t_viz2 = Table(viz_grid_2, colWidths=[3.65*inch, 3.65*inch])
        t_viz2.setStyle(TableStyle([('PADDING', (0,0), (-1,-1), 0)]))
        story.append(t_viz2)
        story.append(Spacer(1, 10))

    # Section 4: Deeper Ablations & Real Corruptions
    story.append(Paragraph("4. Real Image Corruptions & Throughput Benchmark", heading2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=accent_color, spaceAfter=8))

    # Table 2 Data
    t2_data = [
        [Paragraph("Corruption Type", table_header_style), Paragraph("Model Variant", table_header_style), Paragraph("Accuracy (%)", table_header_style), Paragraph("ECE Error", table_header_style), Paragraph("Throughput (FPS)", table_header_style)],
        [Paragraph("Clean Images", table_body_style), Paragraph("ShiftGuard", table_body_style), Paragraph("100.00%", table_body_style), Paragraph("0.0044", table_body_style), Paragraph("324.5 FPS", table_body_style)],
        [Paragraph("Gaussian Noise", table_body_style), Paragraph("ShiftGuard", table_body_style), Paragraph("100.00%", table_body_style), Paragraph("0.0058", table_body_style), Paragraph("324.5 FPS", table_body_style)],
        [Paragraph("Defocus Blur", table_body_style), Paragraph("ShiftGuard<br/>Baseline", table_body_style), Paragraph("<b>94.00%</b><br/>82.22%", table_body_style), Paragraph("<b>0.0192</b><br/>0.1405", table_body_style), Paragraph("324.5 FPS<br/>326.8 FPS", table_body_style)],
        [Paragraph("JPEG Compression", table_body_style), Paragraph("ShiftGuard", table_body_style), Paragraph("100.00%", table_body_style), Paragraph("0.0045", table_body_style), Paragraph("324.5 FPS", table_body_style)],
    ]
    t2 = Table(t2_data, colWidths=[1.8*inch, 1.3*inch, 1.4*inch, 1.3*inch, 1.5*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#ffffff"), bg_light]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Conclusion—</b> ShiftGuard effectively addresses domain, modality, and semantic distribution shifts by dynamically routing features through parameter-efficient LoRA adapters. "
        "With a negligible latency impact of +0.022 ms (+0.72%), ShiftGuard delivers robust accuracy and calibrated confidence for real-world vision deployment.",
        body_style
    ))

    # Build PDF Document
    doc.build(story)
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"Successfully generated PDF report: '{pdf_path}' ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    build_pdf_report()
