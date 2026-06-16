import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv

# Load configuration
load_dotenv()

def send_email(subject, html_content, chart_path=None):
    """
    Sends an HTML email with an optional inline chart image using SMTP.
    """
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT")
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    recipient_email = os.getenv("RECIPIENT_EMAIL")

    # Validate configuration
    missing_vars = []
    if not smtp_server: missing_vars.append("SMTP_SERVER")
    if not smtp_port: missing_vars.append("SMTP_PORT")
    if not sender_email or sender_email == "your_sending_email@gmail.com": missing_vars.append("SENDER_EMAIL")
    if not sender_password or sender_password == "your_email_app_password": missing_vars.append("SENDER_PASSWORD")
    if not recipient_email or recipient_email == "your_receiving_email@gmail.com": missing_vars.append("RECIPIENT_EMAIL")

    if missing_vars:
        print(f"Warning: Email configuration is missing or holds placeholder values: {', '.join(missing_vars)}")
        print("To send actual emails, please fill out these parameters in your .env file.")
        return False

    try:
        port = int(smtp_port)
    except ValueError:
        print(f"Error: SMTP_PORT must be an integer, got: {smtp_port}")
        return False

    # Initialize the email message
    # If we are embedding an inline image, we must use multipart/related
    if chart_path and os.path.exists(chart_path):
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email
        
        # Create an alternative subpart for text/html fallback
        msg_alternative = MIMEMultipart("alternative")
        msg.attach(msg_alternative)
        
        # Attach the HTML body to alternative section
        part_html = MIMEText(html_content, "html")
        msg_alternative.attach(part_html)
        
        # Attach the image to the main related container with matching Content-ID
        try:
            with open(chart_path, "rb") as f:
                img_data = f.read()
            msg_image = MIMEImage(img_data)
            # Content-ID matches the src="cid:comparison_chart" in the HTML template
            msg_image.add_header("Content-ID", "<comparison_chart>")
            msg_image.add_header("Content-Disposition", "inline", filename=os.path.basename(chart_path))
            msg.attach(msg_image)
            print(f"Attached inline chart image: {chart_path}")
        except Exception as e:
            print(f"Warning: Failed to attach image to email: {e}")
    else:
        # Standard fallback if no image is attached
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email
        
        part_html = MIMEText(html_content, "html")
        msg.attach(part_html)

    # SMTP Transmission block
    try:
        if port == 465:
            print(f"Connecting to SMTP server {smtp_server}:{port} via SSL...")
            with smtplib.SMTP_SSL(smtp_server, port) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, msg.as_string())
        else:
            print(f"Connecting to SMTP server {smtp_server}:{port} via STARTTLS...")
            with smtplib.SMTP(smtp_server, port) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, msg.as_string())
                
        print(f"Email sent successfully to {recipient_email}!")
        return True

    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
