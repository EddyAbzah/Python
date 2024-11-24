import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE
from email import encoders

subject = 'Termination of Employment'
file_name = 'file_name.txt'
file_path = r"E:\PycharmProjects\send_email\מכתב סיום העסקה.pdf"
email_from = 'zvi.lando@solaredge.com'
email_to = 'pavel.h@solaredge.com'

msg = MIMEMultipart()
msg['From'] = email_from
msg['To'] = COMMASPACE.join([email_to])
msg['Subject'] = subject

body = """
Dear Pavel,

I hope this email finds you well. After careful consideration and review of recent events, I regret to inform you that we must terminate your employment with Solaredge, effective immediately.
The decision stems from the critical incident involving the failure to connect the battery to the inverter, despite the established protocols and multiple trainings provided. This oversight significantly impacted our operations and posed a potential safety hazard, which we cannot overlook.

We value the contributions you have made during your time with us, but consistency in adhering to operational standards is paramount to ensuring the safety and efficiency of our work environment.
Your final paycheck, including any accrued benefits, will be processed and sent to you by [specific date]. Additionally, please arrange to return any company property by [specific date]. Should you have any questions regarding your termination or final compensation, feel free to reach out to ido.d.

We wish you the best in your future endeavors.


Sincerely,
Zivi Lando
Advisor to the CEO
Solaredge Technologies
"""

msg.attach(MIMEText(body, 'plain'))
part = MIMEBase('application', "octet-stream")
part.set_payload(open(file_path, "rb").read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', 'attachment', filename=file_name)
msg.attach(part)
smtpObj = smtplib.SMTP('mail.solaredge.com')
smtpObj.sendmail(email_from, email_to, msg.as_string())
smtpObj.quit()
