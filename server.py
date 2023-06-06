import os
import sys

import cgi
import cgitb
from html import escape
import base64

include_path = '/var/webcam_project/www'

cgitb.enable(display=0, logdir=f"""{include_path}/tmp_errors""") # include_path is OUTDIR

sys.path.insert(0, include_path)

def enc_print(string='', encoding='utf8'):
    sys.stdout.buffer.write(string.encode(encoding) + b'\n')



args = cgi.FieldStorage()

chunk = '' if not args.getvalue( "chunk" ) else escape( args.getvalue( "chunk" ) )

mp4 = 'webcam.mp4'

with open (mp4, 'ab') as f:
    f.write( base64.b64decode(chunk) )

html = 'success'

enc_print("Content-Type:text/html;charset=utf-8;")
enc_print()        
enc_print(html)