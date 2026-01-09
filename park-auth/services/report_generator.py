"""
Report Generation Service for Park Activity Monitoring
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import os
import config
import cv2
import numpy as np
from typing import Dict, List, Tuple
import tempfile


class ReportGenerator:
    """Generate PDF reports for detection results"""
    
    def __init__(self):
        """Initialize report generator"""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Alert style
        self.styles.add(ParagraphStyle(
            name='Alert',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))
        
        # Success style
        self.styles.add(ParagraphStyle(
            name='Success',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.green,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))
    
    def generate_image_report(self, 
                             filename: str,
                             detections: Dict,
                             annotated_image: np.ndarray,
                             output_path: str = None) -> str:
        """
        Generate a PDF report for image detection
        
        Args:
            filename: Name of the analyzed file
            detections: Detection dictionary
            annotated_image: Annotated image with bounding boxes
            output_path: Optional custom output path
            
        Returns:
            Path to the generated PDF report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(config.OUTPUT_DIR, f"report_{timestamp}.pdf")
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph("🏞️ Park Activity Monitoring Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Report metadata
        metadata = [
            ['Report Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['File Analyzed:', filename],
            ['File Type:', 'Image'],
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Detection Summary
        story.append(Paragraph("Detection Summary", self.styles['CustomHeading']))
        
        total_count = detections.get('total_count', 0)
        authorized_count = detections.get('authorized_count', 0)
        unauthorized_count = detections.get('unauthorized_count', 0)
        
        summary_data = [
            ['Metric', 'Count'],
            ['Total Detections', str(total_count)],
            ['✅ Authorized Activities', str(authorized_count)],
            ['⚠️ Unauthorized Activities', str(unauthorized_count)],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Alert message
        if unauthorized_count > 0:
            alert_text = f"🚨 ALERT: {unauthorized_count} unauthorized activity(ies) detected!"
            story.append(Paragraph(alert_text, self.styles['Alert']))
        else:
            success_text = "✅ All detected activities are authorized!"
            story.append(Paragraph(success_text, self.styles['Success']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Annotated Image
        story.append(Paragraph("Annotated Image", self.styles['CustomHeading']))
        
        # Save annotated image temporarily
        temp_img_path = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(temp_img_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        # Add image to report
        img = Image(temp_img_path, width=6*inch, height=4*inch, kind='proportional')
        story.append(img)
        story.append(Spacer(1, 0.3*inch))
        
        # Detailed Detections
        if total_count > 0:
            story.append(Paragraph("Detailed Detection Information", self.styles['CustomHeading']))
            
            detection_data = [['#', 'Class', 'Confidence']]
            
            # Load detector to check class names
            from services.yolo_detector import YOLODetector
            detector = YOLODetector()
            
            for idx, (cls, conf) in enumerate(zip(
                detections.get('classes', []),
                detections.get('confidences', [])
            )):
                cls_name = detector.get_class_name(int(cls))
                is_unauthorized = detector.is_class_unauthorized(int(cls))
                class_label = f"⚠️ {cls_name}" if is_unauthorized else f"✅ {cls_name}"
                confidence_pct = f"{conf * 100:.1f}%"
                detection_data.append([str(idx + 1), class_label, confidence_pct])
            
            detection_table = Table(detection_data, colWidths=[0.5*inch, 3*inch, 2*inch])
            detection_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            story.append(detection_table)
        
        # Build PDF
        doc.build(story)
        
        # Clean up temp image
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
        return output_path
    
    def generate_video_report(self,
                             filename: str,
                             total_frames: int,
                             unauthorized_frames: List[Tuple[int, np.ndarray, Dict]],
                             output_path: str = None) -> str:
        """
        Generate a PDF report for video detection
        
        Args:
            filename: Name of the analyzed video file
            total_frames: Total number of frames in the video
            unauthorized_frames: List of (frame_num, frame_img, detections) tuples
            output_path: Optional custom output path
            
        Returns:
            Path to the generated PDF report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(config.OUTPUT_DIR, f"video_report_{timestamp}.pdf")
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph("🏞️ Park Activity Monitoring Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Report metadata
        metadata = [
            ['Report Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['File Analyzed:', filename],
            ['File Type:', 'Video'],
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Video Analysis Summary
        story.append(Paragraph("Video Analysis Summary", self.styles['CustomHeading']))
        
        unauthorized_frame_count = len(unauthorized_frames)
        percentage = (unauthorized_frame_count / total_frames * 100) if total_frames > 0 else 0
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Frames Analyzed', str(total_frames)],
            ['⚠️ Frames with Unauthorized Activity', str(unauthorized_frame_count)],
            ['Percentage', f"{percentage:.1f}%"],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Alert message
        if unauthorized_frame_count > 0:
            alert_text = f"🚨 ALERT: Unauthorized activities detected in {unauthorized_frame_count} frame(s)!"
            story.append(Paragraph(alert_text, self.styles['Alert']))
        else:
            success_text = "✅ No unauthorized activities detected in the video!"
            story.append(Paragraph(success_text, self.styles['Success']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Unauthorized Frames Details
        if unauthorized_frame_count > 0:
            story.append(Paragraph("Unauthorized Activity Frames", self.styles['CustomHeading']))
            
            # Limit to first 10 frames in report to avoid huge PDFs
            max_frames_in_report = 10
            frames_to_show = unauthorized_frames[:max_frames_in_report]
            
            # Keep track of temp files to delete after PDF is built
            temp_files = []
            
            for idx, (frame_num, frame_img, detections) in enumerate(frames_to_show):
                story.append(Paragraph(f"Frame #{frame_num}", self.styles['Heading3']))
                
                # Frame statistics
                frame_stats = [
                    ['Unauthorized Count:', str(detections.get('unauthorized_count', 0))],
                    ['Total Detections:', str(detections.get('total_count', 0))],
                ]
                
                stats_table = Table(frame_stats, colWidths=[2*inch, 2*inch])
                stats_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(stats_table)
                story.append(Spacer(1, 0.1*inch))
                
                # Save frame temporarily
                temp_img_path = tempfile.mktemp(suffix='.jpg')
                cv2.imwrite(temp_img_path, cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))
                temp_files.append(temp_img_path)  # Track for later cleanup
                
                # Add frame image
                img = Image(temp_img_path, width=5*inch, height=3*inch, kind='proportional')
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
                
                # Add page break after every 2 frames
                if (idx + 1) % 2 == 0 and idx < len(frames_to_show) - 1:
                    story.append(PageBreak())
            
            if unauthorized_frame_count > max_frames_in_report:
                note = f"Note: Showing first {max_frames_in_report} of {unauthorized_frame_count} frames with unauthorized activities."
                story.append(Paragraph(note, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Clean up temp files AFTER PDF is built
        if unauthorized_frame_count > 0:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return output_path
