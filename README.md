# claude-skills

Claude에서 쓰는 스킬 모음입니다. 폴더 하나가 스킬 하나이고, 각 폴더의 `SKILL.md` 가 스킬 본체입니다.

| 스킬 | 하는 일 |
|---|---|
| [keyword-news-montage](keyword-news-montage/) | 주제와 단어를 주면, 그 단어가 화면 가운데 형광 하이라이트로 고정된 채 주제 관련 한글 신문 지면이 빠르게 바뀌고 숫자 카운터와 낱자·도장 엔딩으로 끝나는 15.5초 mp4(1440×1080)를 만든다. |

## keyword-news-montage 직접 돌리기

필요한 것: Python 3, ffmpeg, 한글 Noto CJK 서체(Noto Sans/Serif CJK KR), `pip install numpy opencv-python-headless pillow`

```bash
cd keyword-news-montage
python3 montage.py examples/hangeulnal.json out/hangeulnal.mp4          # 본 렌더
python3 montage.py examples/hangeulnal.json out/hangeulnal.mp4 --sheet  # 검수 시트(정지 화면 30장)
```

설정 파일 쓰는 법은 `keyword-news-montage/SKILL.md` 의 «config.json 쓰는 법» 에 있습니다.
