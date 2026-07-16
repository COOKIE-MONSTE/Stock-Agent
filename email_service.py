import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv

# Load configuration
load_dotenv()

def send_email(subject, html_content, chart_path=None, logo_path=None):
    """
    Sends an HTML email with optional inline images (logo + chart), each
    embedded via Content-ID so they render without the recipient having to
    click "load images".

    Inline images are passed as (path, content_id) pairs; the content_id must
    match the src="cid:..." reference in the HTML. The logo uses cid:brand_logo
    and the chart uses cid:comparison_chart.
    """
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT")
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    recipient_email = os.getenv("RECIPIENT_EMAIL")
    recipient_list = [r.strip() for r in recipient_email.split(",")] if recipient_email else []

    # Validate configuration
    missing_vars = []
    if not smtp_server: missing_vars.append("SMTP_SERVER")
    if not smtp_port: missing_vars.append("SMTP_PORT")
    if not sender_email or sender_email == "your_sending_email@gmail.com": missing_vars.append("SENDER_EMAIL")
    if not sender_password or sender_password == "your_email_app_password": missing_vars.append("SENDER_PASSWORD")
    if not recipient_list or recipient_email == "your_receiving_email@gmail.com": missing_vars.append("RECIPIENT_EMAIL")

    if missing_vars:
        print(f"Warning: Email configuration is missing or holds placeholder values: {', '.join(missing_vars)}")
        print("To send actual emails, please fill out these parameters in your .env file.")
        return False

    try:
        port = int(smtp_port)
    except ValueError:
        print(f"Error: SMTP_PORT must be an integer, got: {smtp_port}")
        return False

    # Collect the inline images we actually have, as (path, content_id) pairs.
    inline_images = []
    if logo_path and os.path.exists(logo_path):
        inline_images.append((logo_path, "brand_logo"))
    if chart_path and os.path.exists(chart_path):
        inline_images.append((chart_path, "comparison_chart"))

    # If we have any inline images, use multipart/related; else plain alternative.
    if inline_images:
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = ", ".join(recipient_list)

        msg_alternative = MIMEMultipart("alternative")
        msg.attach(msg_alternative)
        part_html = MIMEText(html_content, "html")
        msg_alternative.attach(part_html)

        for img_path, cid in inline_images:
            try:
                with open(img_path, "rb") as f:
                    img_data = f.read()
                msg_image = MIMEImage(img_data)
                msg_image.add_header("Content-ID", f"<{cid}>")
                msg_image.add_header("Content-Disposition", "inline",
                                     filename=os.path.basename(img_path))
                msg.attach(msg_image)
                print(f"Attached inline image '{cid}': {img_path}")
            except Exception as e:
                print(f"Warning: Failed to attach image {img_path} ({cid}): {e}")
    else:
        # Standard fallback if no images are attached
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = ", ".join(recipient_list)

        part_html = MIMEText(html_content, "html")
        msg.attach(part_html)

    # SMTP Transmission block
    try:
        if port == 465:
            print(f"Connecting to SMTP server {smtp_server}:{port} via SSL...")
            with smtplib.SMTP_SSL(smtp_server, port) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_list, msg.as_string())
        else:
            print(f"Connecting to SMTP server {smtp_server}:{port} via STARTTLS...")
            with smtplib.SMTP(smtp_server, port) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_list, msg.as_string())
                
        print(f"Email sent successfully to {', '.join(recipient_list)}!")
        return True

    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
        
