# Phase 0 Failure Mode Triage

_n=11 queries with V0a weighted < 0.6_

## Mode Distribution
- **gold_absent**: 4
- **korean_miss**: 4
- **wrong_field**: 1
- **dupe_paper**: 1
- **low_info_query**: 1

## Query Detail
| query_id | bucket | lang | mode | diagnosis |
|----------|--------|------|------|-----------|
| synth_ko_a65fec21 | Concept | ko | `wrong_field` | 검색된 사실 전체가 탄성 폴리머 액추에이터 관련 논문에서 나온 것으로, 그래핀 CVD 성장 메커니즘과 무관한 완전히 다른 분야의 문서가 상위 결 |
| synth_en_906d117f | Concept | en | `gold_absent` | No facts were returned at all, indicating that papers on gallium-based room-temp |
| synth_en_79113e36 | Mechanism | en | `gold_absent` | No facts were returned because the KB (focused on polymer actuator performance)  |
| synth_ko_a92dace3 | Metric | ko | `korean_miss` | 한국어 쿼리('고전압 측 입력 전압')가 SST 관련 영문 fact [5]를 임베딩 공간에서 매칭하지 못해 전체 스코어가 0.000으로 수렴했으 |
| synth_ko_01807ef7 | Metric | ko | `korean_miss` | 쿼리의 한국어 전문 용어(탄소나노튜브, 전열 구동기, 구부림 곡률)가 영문 인덱스의 해당 개념(MWCNT, thermal actuator, be |
| synth_ko_970ace42 | Metric | ko | `korean_miss` | Korean 쿼리 '유전상수'가 인덱스 내 영문 'dielectric constant' 사실과 임베딩 공간에서 정렬되지 않아 유사도 점수가 전부 |
| synth_en_58ae6a31 | Metric | en | `gold_absent` | No facts were returned at all, indicating the paper containing this specific EGD |
| synth_ko_cf0c95fb | Metric | ko | `korean_miss` | 한국어 쿼리의 '녹는점 저하'가 인덱스에 저장된 영어 용어('melting point depression', 'size-dependent mel |
| synth_en_8bc697c7 | Mechanism | en | `gold_absent` | No facts were returned at all, indicating that papers specifically covering the  |
| synth_ko_c50237c6 | Concept | ko | `dupe_paper` | 5개 retrieved facts 전부 동일한 dielectric elastomer 논문(Elsevier 1998)에서 나온 것으로, 쿼리의 ' |
| synth_en_90305b16 | Metric | en | `low_info_query` | 'Weak force' is ambiguous (could mean weak nuclear force, low-magnitude mechanic |

## Raw

### synth_ko_a65fec21 [Concept/ko]
**Query**: 구리와 니켈 표면에서 그래핀의 층수 형성이 다른 이유는 무엇인가?
**Mode**: `wrong_field`
**Diagnosis**: 검색된 사실 전체가 탄성 폴리머 액추에이터 관련 논문에서 나온 것으로, 그래핀 CVD 성장 메커니즘과 무관한 완전히 다른 분야의 문서가 상위 결과를 점령했습니다.
**Top retrieved fact**: The electrostriction of elastomeric polymer dielectrics with compliant electrodes is potentially useful as a small-scale

### synth_en_906d117f [Concept/en]
**Query**: What room-temperature liquid metal is commonly used in flexible and stretchable electronic applications?
**Mode**: `gold_absent`
**Diagnosis**: No facts were returned at all, indicating that papers on gallium-based room-temperature liquid metals (EGaIn, Galinstan) for stretchable electronics have not been ingested into this KB, which is focused on polymer/actuator materials.

### synth_en_79113e36 [Mechanism/en]
**Query**: What mechanism causes crack propagation in polymer dielectrics under electrostatic stress?
**Mode**: `gold_absent`
**Diagnosis**: No facts were returned because the KB (focused on polymer actuator performance) likely contains no ingested papers specifically addressing crack propagation mechanisms in dielectric polymers under electrostatic stress — a dielectric failure mechanics topic outside the vault's coverage.

### synth_ko_a92dace3 [Metric/ko]
**Query**: SST 컨버터의 고전압 측 입력 전압은 얼마인가?
**Mode**: `korean_miss`
**Diagnosis**: 한국어 쿼리('고전압 측 입력 전압')가 SST 관련 영문 fact [5]를 임베딩 공간에서 매칭하지 못해 전체 스코어가 0.000으로 수렴했으며, 상위 결과가 'crosstalk'라는 표면 토큰 유사도로 인한 노이즈 fact에 점유되었다.
**Top retrieved fact**: However, challenges like sound crosstalk between exciters and non-uniform frequency responses often compromise audio qua

### synth_ko_01807ef7 [Metric/ko]
**Query**: 다층 벽면 탄소나노튜브/고분자 복합재 전열 구동기가 달성할 수 있는 구부림 곡률은 얼마인가?
**Mode**: `korean_miss`
**Diagnosis**: 쿼리의 한국어 전문 용어(탄소나노튜브, 전열 구동기, 구부림 곡률)가 영문 인덱스의 해당 개념(MWCNT, thermal actuator, bending curvature)과 임베딩 공간에서 매칭되지 않아 모든 점수가 0.000이며, 관계없는 유전 탄성체 문헌이 fallback으로 반환되었음.
**Top retrieved fact**: The electrostriction of elastomeric polymer dielectrics with compliant electrodes is potentially useful as a small-scale

### synth_ko_970ace42 [Metric/ko]
**Query**: 실리콘 탄성체의 유전상수는 얼마인가?
**Mode**: `korean_miss`
**Diagnosis**: Korean 쿼리 '유전상수'가 인덱스 내 영문 'dielectric constant' 사실과 임베딩 공간에서 정렬되지 않아 유사도 점수가 전부 0.000으로 떨어지고, 관련 수치 사실 대신 같은 도메인의 일반 서술문이 폴백으로 반환됐다.
**Top retrieved fact**: The electrostriction of elastomeric polymer dielectrics with compliant electrodes is potentially useful as a small-scale

### synth_en_58ae6a31 [Metric/en]
**Query**: What was the EGDMA crosslinker loading relative to AAm in the polymer formulation?
**Mode**: `gold_absent`
**Diagnosis**: No facts were returned at all, indicating the paper containing this specific EGDMA/AAm crosslinker formulation detail was either not ingested or its quantitative synthesis conditions were not extracted into the KB.

### synth_ko_cf0c95fb [Metric/ko]
**Query**: 10nm 이하의 나노구조에서 녹는점이 얼마나 저하되는가?
**Mode**: `korean_miss`
**Diagnosis**: 한국어 쿼리의 '녹는점 저하'가 인덱스에 저장된 영어 용어('melting point depression', 'size-dependent melting')와 매칭되지 않아 결과가 0건 반환됨.

### synth_en_8bc697c7 [Mechanism/en]
**Query**: What is the positive feedback mechanism by which the electromechanical instability in dielectric elastomers leads to electrical breakdown?
**Mode**: `gold_absent`
**Diagnosis**: No facts were returned at all, indicating that papers specifically covering the electromechanical instability positive-feedback breakdown mechanism in dielectric elastomers (e.g., Zhao & Suo 2007, Koh et al.) have not been ingested into the knowledge base.

### synth_ko_c50237c6 [Concept/ko]
**Query**: 협력형 마이크로 액추에이터 어레이는 어떤 마이크로 유체 시스템을 가능하게 하는가?
**Mode**: `dupe_paper`
**Diagnosis**: 5개 retrieved facts 전부 동일한 dielectric elastomer 논문(Elsevier 1998)에서 나온 것으로, 쿼리의 '액추에이터' 키워드가 해당 논문을 과대 대표하게 만들어 microfluidic cooperative arrays 관련 다른 논문들이 완전히 배제됨.
**Top retrieved fact**: The electrostriction of elastomeric polymer dielectrics with compliant electrodes is potentially useful as a small-scale

### synth_en_90305b16 [Metric/en]
**Query**: What is the force threshold for weak force detection?
**Mode**: `low_info_query`
**Diagnosis**: 'Weak force' is ambiguous (could mean weak nuclear force, low-magnitude mechanical force, or a sensor detection threshold), so the query lacks sufficient specificity to match how force-detection thresholds are phrased in ingested materials-science facts.
