import email.utils
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

subject = 'כמה מילים ממני אלייך'
file_name = 'file_name.txt'
file_path = r"C:\Users\eddy.a\OneDrive - SolarEdge\Desktop\mana.txt"
mail_from = 'zvi.lando@solaredge.com'
mail_to = 'dekel.k@solaredge.com'
password = ''
msg = MIMEMultipart()
msg['From'] = mail_from
msg['To'] = email.utils.COMMASPACE.join([mail_to])
msg['subject'] = subject

body = """תגידי לדהן שהוא שרמוטה

המצב בו אנו פועלים, מביא עימו בכל שבוע אתגרים חדשים, ולצידם אני מקווה, אפשר גם לחזור לחוות רגעים קטנים של רגיעה ושקט.

אני שמח שמשרתי ומשרתות המילואים שלנו ממשיכים לשמור על קשר, מגיעים מדי פעם לביקור בבית, וחלקכם/ן אף כבר שוחררתם וחזרתם בינתיים לשגרה.
אני גאה בעובדות ובעובדים שלנו, אשר בני ובנות זוגם משרתים במילואים, ומצליחות/ים לקיים "שגרת חירום" בבתיהם.

עם זאת, הוחלט על צמצומים נרחבים, ובזאת צר לי לבשר לך אדוני הנכבד
אתה מפוטר לאלתר

קח את כל החפצים שלך, שים בארגז, וקפוץ מקומה 7 מצידי כי גמרת עליי
"""

msg.attach(MIMEText(body, 'plain'))
part = MIMEBase('application', "octet-stream")
part.set_payload(open(file_path, "rb").read())
encoders.encode_base64(part)
if file_name is None or file_name == '':
    part.add_header('Content-Disposition', 'attachment; file_name="{}"'.format(os.path.basename(file_path)))
else:
    part.add_header('Content-Disposition', 'attachment', file_name=file_name)  # or

msg.attach(part)
smtpObj = smtplib.SMTP('mail.solaredge.com')
if password is not None and password != '':
    smtpObj.login(mail_from, password)
smtpObj.sendmail(mail_from, mail_to, msg.as_string())
smtpObj.quit()
