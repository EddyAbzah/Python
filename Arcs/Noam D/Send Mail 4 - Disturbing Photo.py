import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import COMMASPACE, make_msgid
import os

# Email details
subject = 'Urgent: Photo on Social Media'
file_name = 'beach photo.jpg'
file_path = r"C:\Users\eddy.a\Downloads\beach photo.jpg"
email_from = 'Rachel.Prishkolnik@solaredge.com'  # Change to your email
email_to = 'Noam.Dahan@solaredge.com, Yvgeni.Naumov@solaredge.com'  # Change to your coworker's email
email_bcc = 'Eddy.Abzah@solaredge.com'  # Your email address in Bcc

# Create the message container
msg = MIMEMultipart('related')
msg['From'] = email_from
msg['To'] = COMMASPACE.join([email_to])
msg['Bcc'] = COMMASPACE.join([email_bcc])
msg['Subject'] = subject

# Email body with HTML content
body = """
<html>
  <body>
    <p>Dear Noam Dahan and Yvgeni Naumov,</p>
    <p>I hope you're doing well.<br>
    I'm reaching out with a matter that requires your immediate attention.<br>
    It has come to my attention that a photo of both of you currently circulating on SolarEdge's social media; and is becoming a subject of concern within the company.<br>
    Given the nature of the image, I thought it was important to let you know, as it might reflect negatively on your public image.<br>
    I’ll be blunt: the image is disturbing to some of our clients.<br>
    Especially, the German clients...</p>

    <p>Here is the offending image:</p>
    <img src="cid:image1" alt="Inappropriate Photo">

    <p>I would recommend removing the photo at your earliest convenience to avoid any potential issues.</p>

    <p>If you'd like to discuss this further or need assistance, feel free to reach out.</p>
    
    <p>Sincerely,<br>
    Rachel Prishkolnik<br>
    VP General Counsel and Corporate Secretary<br>
    Solaredge Technologies</p>
  </body>
</html>
"""

msg.attach(MIMEText(body, 'html'))

# Open the image file and attach it as an embedded image
with open(file_path, 'rb') as f:
    img = MIMEImage(f.read())
    img.add_header('Content-ID', '<image1>')  # Use the same CID referenced in the HTML part
    img.add_header('Content-Disposition', 'inline', filename=file_name)
    msg.attach(img)

# Send the email
try:
    with smtplib.SMTP('mail.solaredge.com') as smtpObj:
        smtpObj.starttls()  # Secure the connection
        if email_bcc != "":
            smtpObj.sendmail(email_from, [email_to] + [email_bcc], msg.as_string())
        else:
            smtpObj.sendmail(email_from, email_to, msg.as_string())  # Send email
        print("Email sent successfully!")
except Exception as e:
    print(f"Error sending email: {e}")
