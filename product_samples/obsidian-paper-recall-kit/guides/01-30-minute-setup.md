# 30분 세팅 가이드

## 준비물

- Obsidian 설치: https://obsidian.md
- 논문 PDF 또는 Zotero 라이브러리
- 오늘 정리할 논문 1편

이 키트는 Obsidian vault 안에서 쓰는 것을 기준으로 만들었습니다. Graph view는 별도
설치가 아니라 Obsidian 기본 기능입니다. 화면에 보이지 않으면 Settings → Core plugins에서
Graph view가 켜져 있는지 확인하세요.

Zotero가 없어도 시작할 수 있습니다. 처음에는 PDF 제목과 핵심 내용을 손으로
옮겨도 됩니다. AI를 쓴다면 전체 요약보다 키워드, 연결 개념, 질문 후보만 받는 편이 좋습니다.

이 가이드의 예시는 재료/로봇 논문이지만, 분야 전용 구조는 아닙니다. 자기 분야의 논문으로
같은 폴더와 템플릿을 쓰면 됩니다.

## 0-5분: 폴더 만들기

Obsidian vault 안에 아래 폴더를 만듭니다.

```text
papers/
concepts/
questions/
meetings/
templates/
```

역할은 단순합니다.

| Folder | Purpose |
| --- | --- |
| `papers/` | 논문별 노트 |
| `concepts/` | 이론, 방법, 사례, 장치, 데이터셋 같은 연구 개념 |
| `questions/` | 선택형: 나중에 다시 찾을 질문 |
| `meetings/` | 선택형: 교수님 미팅, 랩미팅, 아이디어 회의 |
| `templates/` | 반복해서 쓸 노트 양식 |

## 5-10분: 템플릿 넣기

`templates/` 폴더 안 파일을 Obsidian vault의 `templates/` 폴더에 넣습니다.

처음에는 `paper-note.md`와 `concept-note.md`만 써도 됩니다. 아래 두 개는 필요할 때 추가합니다.

- `paper-note.md`
- `concept-note.md`
- `question-note.md` optional
- `meeting-note.md` optional

## 10-20분: 첫 논문 노트 만들기

오늘 읽은 논문 1편을 고릅니다. 이미 읽은 논문이어도 괜찮습니다.

`templates/paper-note.md`를 복사해서 `papers/` 안에 새 파일로 만듭니다.

파일명은 너무 예쁘게 만들 필요 없습니다. 다만 나중에 보기 쉽게 아래처럼 씁니다.

```text
2020-christianson-lce-underwater-soft-robot.md
```

추천 형식:

```text
연도-첫저자-핵심키워드.md
```

파일명과 첫 초안은 AI에게 맡겨도 됩니다.

```text
이 논문의 제목, 저자, 연도, 초록을 보고 Obsidian 파일명 후보를 3개 추천해줘.
형식은 연도-첫저자-핵심키워드.md로 해줘.
그리고 한 줄 기억 후보도 3개 써줘.
```

## 20-25분: 개념 노트 하나 연결하기

논문에서 가장 중요한 개념 하나를 고릅니다. 잘 모르겠으면 AI에게 후보를 좁혀 달라고 합니다.

예:

- `Liquid Crystal Elastomer`
- `Soft Robot`
- `Thermal Actuation`

다른 분야라면 `Scaling Law`, `Gene Delivery`, `Gig Economy`처럼 바꿔 쓰면 됩니다.

`concepts/` 안에 개념 노트를 만들고, 논문 노트에서 `[[Liquid Crystal Elastomer]]`처럼
연결합니다.

```text
이 논문에서 개념 노트로 만들 만한 항목 3개를 추천해줘.
각 항목을 이론, 방법, 사례, 장치, 데이터셋, 현상 중 하나로 분류해줘.
너무 넓은 단어는 피하고 나중에 검색에 도움이 되는 개념으로 골라줘.
```

## 25-30분: 질문 노트가 필요한지 보기

미팅에서 다시 물어볼 법하거나, 논문 제목 대신 질문으로 다시 찾을 것 같은 경우에만
질문 노트를 하나 만듭니다.

예:

```text
LCE로 underwater robot을 만든 논문은?
```

질문 노트에는 답을 길게 쓰지 않아도 됩니다. 다시 열어볼 논문 링크만 있어도 됩니다.
필요 없으면 첫날에는 만들지 않아도 됩니다.

AI에게 질문 후보만 받아도 됩니다.

```text
이 논문을 제목 없이 다시 찾는다고 가정하고,
내가 검색할 법한 질문 제목 5개를 만들어줘.
각 질문은 Obsidian 노트 제목으로 쓸 수 있게 짧게 써줘.
```

## 첫 세팅이 끝났을 때

아래 두 파일이 있으면 성공입니다. 질문 파일은 선택입니다.

```text
papers/2020-christianson-lce-underwater-soft-robot.md
concepts/Liquid Crystal Elastomer.md
questions/LCE로 underwater robot을 만든 논문은.md  # optional
```

첫날의 목표는 체계를 완성하는 것이 아니라, 논문 1편이 다시 찾아지는 구조를 만드는 것입니다.
