"""SMTP 邮件发送器（QQ邮箱 SSL）。"""
from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from loguru import logger

from config.settings import NotifyConfig


def send_report(html_body: str, subject: str = "ETF 三因子监测报告") -> bool:
    """发送 HTML 邮件报告。

    返回 True 表示发送成功，失败返回 False（不抛异常）。
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = NotifyConfig.EMAIL_FROM
    msg["To"] = NotifyConfig.EMAIL_TO
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL(NotifyConfig.SMTP_HOST, NotifyConfig.SMTP_PORT) as server:
            server.login(NotifyConfig.SMTP_USER, NotifyConfig.SMTP_PASS)
            server.sendmail(NotifyConfig.EMAIL_FROM, [NotifyConfig.EMAIL_TO], msg.as_string())
        logger.info(f"邮件已发送: {subject}")
        return True
    except Exception as e:
        logger.error(f"邮件发送失败: {e}")
        return False
