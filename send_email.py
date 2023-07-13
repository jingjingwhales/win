import smtplib, os
from email.mime.text import MIMEText
# from datetime import datetime
import pandas as pd
from collections import defaultdict

DEV_EMAIL_LIST = "huayong.chow@gmail.com"
POS_LIST = {"BTC", "ETH", "AVAX", "BNB", "LTC", "ICP", "DOGE",
            "PEOPLE", "TVK", "SUSHI", "DYDX", "SOL", "ADA", "APT", "UNI", "FIL", "MINA"}
POS_PROB = defaultdict(lambda: "")
PROB_THRESHOLD = [-2.0/100, -1.5/100, -1.0/100, -0.5/100, 0.0/100,
                0.5/100, 1.0/100, 1.5/100, 2.0/100,]
FIXED_LEN = 25 - 6
def sendEmail(sender, recipients, subject, content):
    # recipients_lst = recipients.split(',')
    msg = MIMEText(content, 'html')
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipients
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("huayong.chow@gmail.com", "uenzukajlcgdeddy")
    server.send_message(msg)
    server.quit()

if __name__ == "__main__":
    pred_filename = ""
    for f in os.listdir("./"):
        if "prediction_from" in f:
            pred_filename = f
            break
    df_FN = pd.read_csv("./FNG_index.index")
    df_FN.sort_values(by=["timestamp"], ascending=False, inplace=True)
    latest_value = df_FN.iloc[0,:]
    fng_value = latest_value["fng_value"]
    fng_classification = latest_value["fng_classification"]

    f_lst = pred_filename.split("_")
    from_dt = f_lst[-3]
    to_dt = f_lst[-1][:-4]
    subject = "Crypto Price Prediction From UTC{from_dt} To UTC{to_dt}".format(from_dt=from_dt, to_dt=to_dt)
    sender = "no_reply"
    recipients = DEV_EMAIL_LIST
    content = ""
    df = pd.read_csv(pred_filename)
    # df.sort_values(by=["price_up_prob"], ascending=False, inplace=True)
    # btc_prob = 0
    # header = "The probability of price goes up:\n\t\t\t"
    # for threshold in PROB_THRESHOLD:
    #     header += "More than {0:.1f}%\t\t\t".format(threshold * 100)
    # content += "\n"
    # for index, row in df.iterrows():
    #     ticker = row["Symbol"]
    #     content += (ticker + ' ' * (FIXED_LEN - len(ticker)))
    #     for threshold in PROB_THRESHOLD:
    #         prob_var = "price_up_{0:.1f}pct_prob_pred".format(threshold * 100)
    #         content += "{0:.2f}%\t\t\t\t\t".format(row[prob_var] * 100)
    #     content += '\n'
    #     if ticker in POS_LIST:
    #         POS_PROB[ticker] += (ticker + ' ' * (FIXED_LEN - len(ticker)))
    #         for threshold in PROB_THRESHOLD:
    #             prob_var = "price_up_{0:.1f}pct_prob_pred".format(threshold * 100)
    #             POS_PROB[ticker] += "{0:.2f}%\t\t\t\t\t".format(row[prob_var] * 100)
    # content = "BTC\t\t\t\t\t{prob_btc}\n".format(prob_btc=POS_PROB["BTC"]) + content
    ######rename the table#########
    col_mapper = {}
    for threshold in PROB_THRESHOLD:
        prob_var = "price_up_{0:.1f}pct_prob_pred".format(threshold * 100)
        df[prob_var] = df[prob_var].apply(lambda val: "{0:.2f}%".format(val * 100))
        col_mapper[prob_var] = "More than {0:.1f}%\t\t\t".format(threshold * 100)
    df.rename(columns=col_mapper, inplace=True)
    df.reset_index(inplace=True)
    df['index'] = df['index'] + 1
    df.rename(columns={"index": "MKT-CAP Rank"}, inplace=True)
    content = """<html>
                    <head>The probability of price goes up:</head>
                    <body>
                    {0}
                    </body>
                </html>
                """.format(df.to_html(index=False))
    sendEmail(sender, recipients, subject, content)

    df_pos = df[df['Symbol'].isin(POS_LIST)]
    pos_content = """<html>
                        <head>The probability of price goes up:</head>
                        <body>
                        {0}
                        </body>
                    </html>
                    """.format(df_pos.to_html(index=False))
    pos_content += "\n\n\nAs of {date}, the BTC market sentiment is {fng_classification}, the greed-fear index value={fng_value}"\
            .format(date=from_dt, fng_classification=fng_classification, fng_value=fng_value)
    sendEmail(sender, "huayong.chow@gmail.com",
              "Crypto Price Prediction for interested only From {from_dt} To {to_dt}".format(from_dt=from_dt, to_dt=to_dt),
              pos_content)
