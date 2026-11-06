import pathlib
ROOT = pathlib.Path(__file__).parent
IM = "../../images/"
posters = [
 # (file, image, position, headline, sub, theme)
 ("01_hero","p1_hero.png","center 40%","Made,<br>not<br>assembled.","The Butcher’s Best. A proper burger, built by hand.","dark"),
 ("02_butcher","p2_butcher.png","center 30%","Meet the<br>mince.","Fresh from a local butcher. Hand-formed. Never frozen.","dark"),
 ("03_bun","p3_bun.png","center 50%","Baked<br>this<br>morning.","Our buns are made from scratch, every day. Not delivered. Baked.","cream"),
 ("04_cheese","p4_cheese.png","center 50%","Local cheese.<br>Proper melt.","Sourced close to home. Melted on top.","cream"),
 ("05_grill","p5_grill.png","center 50%","Nothing frozen.<br>Nothing hidden.","Fresh ingredients. Honest cooking. Everything made here.","dark"),
 ("06_bite","p6_bite.png","center 40%","Take a<br>bite.","The Butcher’s Best. Made, not assembled.","dark"),
 ("07_type",None,None,"Beef.<br>Bun.<br>Cheese.","All of it made here, by hand. Nothing off a lorry.","red"),
 ("08_chains","p1_hero.png","center 70%","The big chains<br>assemble.<br>We make.","Real butcher’s beef. Homemade bun. The Butcher’s Best.","cream"),
 ("09_trio","p2_butcher.png","center 60%","Your butcher.<br>Your baker.<br>Your burger.","Local people. Local ingredients. One proper burger.","dark"),
 ("10_launch","p1_hero.png","center 45%","Now<br>serving.","The Butcher’s Best<br>[Address] · [Opening date] · [Hours] · [@handle]","dark"),
]
CSS = """
*{box-sizing:border-box;margin:0}
body{width:1080px;height:1350px;position:relative;overflow:hidden;font-family:Georgia,serif;background:#1a1410;color:#f5ecd9}
.img{position:absolute;inset:0;background-size:cover}
.shade{position:absolute;inset:0;background:linear-gradient(180deg,rgba(18,12,8,.15) 0%,rgba(18,12,8,.25) 35%,rgba(18,12,8,.92) 100%)}
.cream .shade{background:linear-gradient(180deg,rgba(245,236,217,.0) 0%,rgba(245,236,217,.0) 40%,rgba(245,236,217,.96) 78%)}
.cream{color:#1a1410}
.red{background:#c8321a;color:#f5ecd9}
.wrap{position:absolute;left:70px;right:70px;bottom:90px}
h1{font-family:Impact,'Arial Narrow',sans-serif;font-weight:400;text-transform:uppercase;font-size:150px;line-height:.95;letter-spacing:1px}
.red h1{font-size:230px;line-height:.9}
p{font-size:38px;line-height:1.3;margin-top:30px;max-width:900px;font-style:italic}
.tag{position:absolute;top:60px;left:70px;font-family:Impact,sans-serif;letter-spacing:5px;font-size:30px;background:#e9a825;color:#1a1410;padding:12px 22px}
.cream .tag{background:#1a1410;color:#f5ecd9}
.red .tag{background:#f5ecd9;color:#c8321a}
.tiny{position:absolute;top:70px;right:70px;font-size:22px;letter-spacing:3px;text-transform:uppercase;opacity:.85}
"""
for f,img,pos,h,sub,theme in posters:
    bg = f'<div class="img" style="background-image:url({IM}{img});background-position:{pos}"></div><div class="shade"></div>' if img else ""
    html = f'<!doctype html><meta charset="utf-8"><style>{CSS}</style><body class="{theme}">{bg}<div class="tag">THE BUTCHER’S BEST</div><div class="tiny">Made, not assembled</div><div class="wrap"><h1>{h}</h1><p>{sub}</p></div></body>'
    (ROOT/"src"/f"{f}.html").write_text(html)
print("ok")
