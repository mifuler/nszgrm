"""DEMO ONLY. Synthetic names to verify all four architectures, not evaluation."""
from __future__ import annotations
from collections import defaultdict
from corpus import split3


def demo(seed=20261117):
    prefixes={'zh_native':['张','李','王','赵','陈','林','刘','周','黄','何'],
              'foreign_phonetic_proxy':['布','克','弗','伊','洛','麦','阿','奥','维','玛'],
              'ja_kanji_proxy':['山','田','川','佐','村','野','藤','井','岛','木']}
    middle={'zh_native':['明','伟','丽','平','芳','建','新','军','红'],
            'foreign_phonetic_proxy':['莱','特','拉','斯','尔','伦','卡','姆','妮'],
            'ja_kanji_proxy':['太','田','一','雄','郎','也','子','美','治']}
    wr=[]
    for lab in prefixes:
        for ch in prefixes[lab]:
            for suffix in middle[lab]:
                for d in ['','安','文']:
                    name=ch+suffix+d
                    wr.append({'name':name,'label':lab,'group':name,'tier':'weak',
                               'split':split3('write:'+name,seed)})
    lg=[]
    demo={'en':['麦','布','克'], 'ru':['伊','维','弗'], 'ja':['山','田','村'],
          'fr':['皮','路','德'], 'es':['罗','加','塞'], 'de':['汉','赫','施'], 'ko':['金','朴','崔']}
    for lab,prefixes in demo.items():
        for ch in prefixes:
            for tail in ['娜','安','玛','伦','特','莱','里','斯','文','芳','儿']:
                for d in ['','雅']:
                    name=ch+tail+d
                    lg.append({'name':name,'label':lab,'group':name,'tier':'proxy',
                               'split':split3('lang:'+name,seed)})
    return wr,lg
