# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + active=""
# <<RandomForest,XGBoost, LightGBM 활용 >> 
# -

# <html><head>
# <meta http-equiv="Content-Type" content="text/html; charset=ks_c_5601-1987">
# <meta name="Generator" content="Microsoft Word 15 (filtered)">
# <style>
# <!--
#  /* Font Definitions */
#  @font-face
# 	{font-family:굴림;
# 	panose-1:2 11 6 0 0 1 1 1 1 1;}
# @font-face
# 	{font-family:"Cambria Math";
# 	panose-1:2 4 5 3 5 4 6 3 2 4;}
# @font-face
# 	{font-family:"맑은 고딕";
# 	panose-1:2 11 5 3 2 0 0 2 0 4;}
# @font-face
# 	{font-family:굴림체;
# 	panose-1:2 11 6 9 0 1 1 1 1 1;}
# @font-face
# 	{font-family:Georgia;
# 	panose-1:2 4 5 2 5 4 5 2 3 3;}
# @font-face
# 	{font-family:inherit;
# 	panose-1:0 0 0 0 0 0 0 0 0 0;}
# @font-face
# 	{font-family:"\@맑은 고딕";
# 	panose-1:2 11 5 3 2 0 0 2 0 4;}
# @font-face
# 	{font-family:"\@굴림";
# 	panose-1:2 11 6 0 0 1 1 1 1 1;}
# @font-face
# 	{font-family:"\@굴림체";
# 	panose-1:2 11 6 9 0 1 1 1 1 1;}
#  /* Style Definitions */
#  p.MsoNormal, li.MsoNormal, div.MsoNormal
# 	{margin-top:0cm;
# 	margin-right:0cm;
# 	margin-bottom:8.0pt;
# 	margin-left:0cm;
# 	text-align:justify;
# 	text-justify:inter-ideograph;
# 	line-height:107%;
# 	text-autospace:none;
# 	word-break:break-hangul;
# 	font-size:10.0pt;
# 	font-family:"맑은 고딕";}
# pre
# 	{mso-style-link:"미리 서식이 지정된 HTML Char";
# 	margin:0cm;
# 	margin-bottom:.0001pt;
# 	font-size:12.0pt;
# 	font-family:굴림체;}
# span.HTMLChar
# 	{mso-style-name:"미리 서식이 지정된 HTML Char";
# 	mso-style-link:"미리 서식이 지정된 HTML";
# 	font-family:굴림체;}
# span.y2iqfc
# 	{mso-style-name:y2iqfc;}
# .MsoChpDefault
# 	{font-family:"맑은 고딕";}
# .MsoPapDefault
# 	{margin-bottom:8.0pt;
# 	text-align:justify;
# 	text-justify:inter-ideograph;
# 	line-height:107%;}
#  /* Page Definitions */
#  @page WordSection1
# 	{size:595.3pt 841.9pt;
# 	margin:3.0cm 72.0pt 72.0pt 72.0pt;}
# div.WordSection1
# 	{page:WordSection1;}
# -->
# </style>
#
# </head>
#
# <body lang="KO">
#
# <div class="WordSection1">
#
# <table width="683" class="MsoNormalTable" style="width:511.95pt;background:white;border-collapse:collapse;border:none" border="1" cellspacing="0" cellpadding="0">
#  <thead>
#   <tr>
#    <td valign="bottom" style="border-top:none;border-left:solid #E6E6E6 1.0pt;&#10;   border-bottom:solid #DDDDDD 1.5pt;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#    <p align="left" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;   text-align:left;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;   word-break:keep-all"><b><span style="font-size:11.5pt;font-family:굴림;&#10;   color:#666666">변수명</span></b></p>
#    </td>
#    <td valign="bottom" style="border-top:none;border-left:solid #E6E6E6 1.0pt;&#10;   border-bottom:solid #DDDDDD 1.5pt;border-right:solid #E6E6E6 1.0pt;&#10;   padding:3.75pt 3.75pt 3.75pt 3.75pt">
#    <p align="left" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;   text-align:left;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;   word-break:keep-all"><b><span style="font-size:11.5pt;font-family:굴림;&#10;   color:#666666">변수</span></b><b><span style='font-size:11.5pt;font-family:&#10;   "Georgia","serif";color:#666666'> </span></b><b><span style="font-size:11.5pt;&#10;   font-family:굴림;color:#666666">설명</span></b></p>
#    </td>
#   </tr>
#  </thead>
#  <tbody><tr>
#   <td valign="top" style="border:none;border-left:solid #E6E6E6 1.0pt;padding:&#10;  3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>class</span></p>
#   </td>
#   <td valign="top" style="border-top:none;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>edible</span><span style="font-size:11.5pt;&#10;  font-family:굴림;color:#666666">식용</span><span lang="EN-US" style='font-size:&#10;  11.5pt;font-family:"Georgia","serif";color:#666666'> = e, poisonous </span><span style="font-size:11.5pt;font-family:굴림;color:#666666">독성</span><span lang="EN-US" style='font-size:11.5pt;font-family:"Georgia","serif";color:#666666'>=
#   p</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>cap-shape</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>bell = b, conical = c, convex = x, flat = f,
#   knobbed = k, sunken = s</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>cap-surface</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>fibrous = f, grooves = g, scaly = y, smooth
#   = s</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>cap-color</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>brown = n, buff = b, cinnamon = c, gray = g,
#   green = r, pink = p, purple = u, red = e, white = w, yellow = y</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>bruises</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>bruises = t, no = f</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>odor</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>almond = a, anise = l, creosote = c, fishy =
#   y, foul = f, musty = m, none = n, pungent = p, spicy = s</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>gill-attachment</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>attached = a, descending = d, free = f,
#   notched = n</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>gill-spacing</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>close = c, crowded = w, distant = d</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>gill-size</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>broad = b, narrow = n</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>gill-color</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>black = k, brown = n, buff = b, chocolate =
#   h, gray = g, green = r, orange = o, pink = p, purple = u, red = e, white = w,
#   yellow = y</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>stalk-shape</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>enlarging = e, tapering = t</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>stalk-root</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>bulbous = b, club = c, cup = u, equal = e,
#   rhizomorphs = z, rooted = r, missing = ?</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>stalk-surface-above-ring</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>fibrous = f, scaly = y, silky = k, smooth =
#   s</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>stalk-surface-below-ring</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>fibrous = f, scaly = y, silky = k, smooth =
#   s</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>stalk-color-above-ring</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>brown = n, buff = b, cinnamon = c, gray = g,
#   orange = o, pink = p, red = e, white = w, yellow = y</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>stalk-color-below-ring</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>brown = n, buff = b, cinnamon = c, gray = g,
#   orange = o,pink = p, red = e, white = w, yellow = y</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>veil-type</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>partial = p, universal = u</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>veil-color</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>brown = n, orange = o, white = w, yellow = y</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>ring-number</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>none = n, one = o, two = t</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>ring-type</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>cobwebby = c, evanescent = e, flaring = f,
#   large = l, none = n, pendant = p, sheathing = s, zone = z</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>spore-print-color</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>black = k, brown = n, buff = b, chocolate =
#   h, green = r, orange =o, purple = u, white = w, yellow = y</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>population</span></p>
#   </td>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:none;border-right:solid #E6E6E6 1.0pt;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>abundant = a, clustered = c, numerous = n,
#   scattered = s, several = v, solitary = y</span></p>
#   </td>
#  </tr>
#  <tr>
#   <td valign="top" style="border-top:solid #DDDDDD 1.0pt;border-left:solid #E6E6E6 1.0pt;&#10;  border-bottom:solid #E6E6E6 1.0pt;border-right:none;padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>habitat</span></p>
#   </td>
#   <td valign="top" style="border:solid #E6E6E6 1.0pt;border-top:solid #DDDDDD 1.0pt;&#10;  padding:3.75pt 3.75pt 3.75pt 3.75pt">
#   <p align="center" class="MsoNormal" style="margin-bottom:0cm;margin-bottom:.0001pt;&#10;  text-align:center;line-height:normal;text-autospace:ideograph-numeric ideograph-other;&#10;  word-break:keep-all"><span lang="EN-US" style='font-size:11.5pt;font-family:&#10;  "Georgia","serif";color:#666666'>grasses = g, leaves = l, meadows = m, paths
#   = p, urban = u, waste = w, woods = d</span></p>
#   </td>
#  </tr>
# </tbody></table>
#
# <p class="MsoNormal"><span lang="EN-US">&nbsp;</span></p>
#
# </div>
#
#
#
#
# </body></html>

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier , StackingClassifier
from sklearn.preprocessing import LabelEncoder

np.random.seed(10)
mr = pd.read_csv('mushroom.csv' , header=None)

mr.iloc[ : , 0].value_counts() # e: 식용 p : 독

# +
# 데이터 기호 -> 숫자 변환

data = [] #x
label = [] #y


for row_index, row in mr.iterrows(): # 8124 , 23
    #row_index = n번째 인덱스 , row = 인덱스값들 시리즈로
    label.append(row.iloc[0]) # y
    row_data=[]

    for v in row.iloc[1:] : #22
        row_data.append(ord(v)) #문자 -=> 아스키코드 변환
    data.append(row_data) # 데이터추가

# -

df_mr = pd.DataFrame(data=data)
df_mr['target'] = label
df_mr

df_mr['target_1'] = LabelEncoder().fit_transform(df_mr['target'])
df_mr

x_train , x_test , y_train , y_test = train_test_split(data , label)

# +
from sklearn.metrics import accuracy_score , classification_report
clf = RandomForestClassifier(random_state=10 , n_jobs=-1).fit(x_train , y_train)
pred = clf.predict(x_test)

score = accuracy_score(y_test , pred)
print(score)

report = classification_report(y_test,pred)
print(report)
# -

from sklearn.tree import export_graphviz

pip install graphviz

from sklearn.tree import export_graphviz
#len(clf.estimators_)
estimator=  clf.estimators_[9]
export_graphviz(estimator,out_file='mushroom01.pdf',  class_names  = np.array(['e','p']) ,
                feature_names= list(mr.columns[1:]), 
                filled=True ,rounded =True  
               )

import graphviz
f = open('mushroom01.pdf') 
dot_graph = f.read()
graphviz.Source(dot_graph)

# 정밀도 : 양성예측도 예측중 맞춘비율
#
# 재현율 : 민감도 TPR 실제중 맞춘비율
#
# ROC : TNR(특이성) : TN / (FP + TN )
#
#  x축을 FPR ( FP / (FP + TN) 1 - TNR
#  
#  y축을 재현율(민감도)
#  
# ACU : ROC 곡선의 면적 1에 가까울수록 좋음|

# +
# 머쉬룸데이터로 XGBOOST / LIGHTGBM 해보기
# -

# ## 파이썬 XGBoost : xgboost ( DMatrix 객체 사용, train()/predict() 함수)

import xgboost as xgb #파이썬거
#파이썬 래퍼 XGBoost : xgboost ( DMatrix 객체 사용, train()/predict() 함수)
from xgboost import plot_importance

data = load_breast_cancer()
x = data.data
y = data.target
cancer_df = pd.DataFrame(data=x , columns=data.feature_names)
cancer_df['target'] = y
cancer_df

x_train , x_test , y_train , y_test = train_test_split(x , y , train_size=0.8  , random_state=0)
x_train.shape

dtrain = xgb.DMatrix(data=x_train , label=y_train)
dtest = xgb.DMatrix(data=x_test , label=y_test)

params = {'max_depth':3 , 'eta': 0.1 , 'objective':'binary:logistic' , 'eval_metric' : 'logloss'}
              #최대깊이      #학습률         #이진?                          #손실함수
wlist = [(dtrain,'train') , (dtest,'eval')]
xgb.model = xgb.train(params=params , dtrain=dtrain,evals = wlist ,num_boost_round = 200)


