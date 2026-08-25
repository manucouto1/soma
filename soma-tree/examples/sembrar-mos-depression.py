"""La investigación de mos-depression-clean, escrita en soma-tree.

No es un ejemplo: es el razonamiento que hay en `docs/PAPER_PLAN.md`,
`docs/EXPERIMENT_MAP.md`, `docs/INTERPRETABILITY_FINDINGS.md` y `RQS.md`, con
sus commits y sus números, puesto en las cinco clases de la capa 2.

Cada intento lleva **las dos versiones**: el commit (el código) y la
configuración (la invocación resuelta, que git no tiene — el mismo commit con
`--decorr-weight 0.1` y con `0.5` son dos experimentos distintos).
"""

import json
import sys
import urllib.request

AT = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7373"


def post(path, body):
    req = urllib.request.Request(
        AT + path,
        data=json.dumps(body).encode() if not isinstance(body, str) else body.encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as said:
        return json.load(said)


def kept(text):
    """Guarda una configuración y devuelve por dónde encontrarla."""
    return post("/api/kept", text)["digest"]


def move(kind, prose, under=(), scope=(), cites=(), course=None):
    body = {"kind": kind, "prose": prose, "under": list(under),
            "scope": list(scope), "cites": list(cites), "who": "manu"}
    if course:
        body["course"] = course
    return post("/api/moves", body)["id"]


def says(frm, to, verb, scope=(), in_part=False):
    post("/api/moves/says", {"from": frm, "to": to, "says": verb,
                             "scope": list(scope), "in_part": in_part})


def code(sha):
    return {"what": "commit", "id": sha}


def config(text):
    """La invocación resuelta: la mitad de la versión que git no tiene."""
    return {"what": "config", "id": kept(text)}


def result(text):
    """Lo que salió. Aparte de la configuración a propósito: con qué se corrió
    y qué midió son dos preguntas, y meterlas en un blob las hace ilegibles
    justo cuando alguien quiere comparar dos intentos.

    Esta investigación no pasó por los ensayos de soma —es anterior—, así
    que sus números no son un `trial` que se pueda recorrer, y decir que lo son
    sería inventar una cita que no apunta a nada. Son lo que se midió, guardado
    igual que la configuración y citado aparte."""
    return {"what": "result", "id": kept(text)}


C = {
    "import": "600de47b4ceb231960565a6cc8a89938a3bd7aec",
    "ext": "63d354acde21d257c9576addd4c6623012f50bac",
    "plan": "8853c1f7130fa6479cb46c22f6dcf4bfd9f052cd",
    "disclosure": "db6192c6d1c8c57d4b0a2c7ee2134cc62c2e714f",
    "grounding": "e675943f2fb9e4f993f14e2f4c40ba0126ac7731",
    "rho": "2861cf7712865b58320278735fe6ddf5a9c13a6d",
    "ladder": "3847d0c1d119652294d30822125398c97fb1503e",
    "rsdd": "ecc25304ca12f7e652538ddfbbee1cdd51a27dab",
    "s3m": "ef91eb9a9821ceaaaebec6b03b4c242208a884bb",
    "transfer": "b5e3a233372e22921aadc8f6477365c4cf5b1852",
    "signal": "f8311866b47faae1868a377d64911adef0912ed4",
    "alif": "16f43a94c772e9aba024bbad2aa3ebfd657f7bc5",
    "grounded": "1d062916a939c97bc3179346463f7207c672bacf",
}

# ── La pregunta con la que se empezó ──

q0 = move("question", "¿Se puede detectar depresión en texto con una arquitectura "
          "que **rinda** y sea **interpretable por construcción**, y no interpretada "
          "a posteriori?")

h0 = move("hypothesis",
          "Un decodificador **puro por síntoma** —un canal por ítem del BDI-II, "
          "readout diagonal por bloques— es interpretable por construcción y "
          "puede alcanzar el rendimiento de un modelo agregador.",
          under=[q0])

# ── Fase 0: el repo limpio no reproducía al sucio ──

q_par = move("question",
             "El repositorio limpio no reproduce el rendimiento del sucio "
             "(`mos_snn_ft` AUROC ≈ 0.95). ¿Qué se perdió al limpiarlo?",
             under=[q0])

a_v1 = move("attempt", "v1_baseline — el ancla, tal cual quedó al limpiar",
            under=[q_par], cites=[code(C["import"]), config(
                "run_tag: v1_baseline\npipelines: todas\nseeds: 5\n"
                "criterio: reproduce los resultados limpios de 2026-04-29\n")])
a_v5 = move("attempt", "v5_multistep_5050 — la pérdida combina `(1-w)·main + w·ms` "
            "en vez de `1·main + w·ms`",
            under=[q_par], cites=[code(C["import"]), config(
                "run_tag: v5_multistep_5050\npipelines: mos_snn_ft, mos_cfc_ft\nseeds: 3\n"
                "cambio: loss combine (1-w)*main + w*ms\n")])
a_v3 = move("attempt", "v3_no_xattn — sin cross-attention de BDI, más cerca del "
            "ganador sucio (DocSeqSelfAttnGRU)",
            under=[q_par], cites=[code(C["import"]), config(
                "run_tag: v3_no_xattn\nuse_bdi_xattn: false\n"
                "pipelines: mos_snn_no_xattn_ft, mos_cfc_no_xattn_ft\nseeds: 3\n")])
a_v4 = move("attempt", "v4_pw_none — `pos_weight=None` en vez de `auto≈3.49`",
            under=[q_par], cites=[code(C["import"]), config(
                "run_tag: v4_pw_none\npos_weight: null\npipelines: mos_snn_ft\nseeds: 3\n")])

f_par = move("finding",
             "CFC se va a 0 y el AUROC deriva respecto al sucio. La diferencia está "
             "en la combinación de pérdidas y en el cross-attention, no en los datos.",
             under=[a_v1])
says(f_par, q_par, "answers", scope=[q_par], in_part=True)

d_par = move("decision",
             "Congelar la primera configuración que recupere AUROC ≈ 0.95 como "
             "`v6_parity_anchor` y correr **todas** las ablaciones de RQ encima. "
             "Hasta entonces cualquier resultado de RQ es sospechoso.",
             under=[f_par], scope=[q_par], course="pursue")

# ── La frontera: qué palanca mueve el readout puro ──

q_front = move("question",
               "¿Qué palanca de precisión levanta el readout **puro**? "
               "Volumen de datos, encoder, preentrenamiento, agregación.",
               under=[h0])

f1 = move("attempt", "F1 · readout puro (base) — el modelo interpretable",
          under=[q_front], cites=[code(C["ladder"]), config(
              "pipeline: mos_csb_indepgru_ft\nrun_tag: csb_paper\nseeds: 5\n"),
              result("cp_f1: 0.576\noracle: 0.616\n")])
f2 = move("attempt", "F2 · GRU compartida (agregadora, con fugas)",
          under=[q_front], cites=[code(C["ladder"]), config(
              "pipeline: mos_csb_gru_ft\nrun_tag: csb_paper\nseeds: 5\n"),
              result("cp_f1: 0.583\noracle: 0.626\n")])
f4 = move("attempt", "F4 · **+ volumen** (RSDD) sobre el readout puro",
          under=[q_front], cites=[code(C["rsdd"]), config(
              "pipeline: mos_csb_indepgru_ft --extra-train rsdd\nrun_tag: csb_rsdd\n"
              "seeds: 5\n"), result("cp_f1: 0.583\noracle: 0.628\n")])
f6 = move("attempt", "F6 · **+ encoder fuerte** (MPNet afinado)",
          under=[q_front], cites=[code(C["ladder"]), config(
              "pipeline: mos_csb_bdimpnet_ft, mos_csb_allmpnet_ft\nrun_tag: csb_paper\n"
              "seeds: 5\n"),
              result("cp_f1: 0.503\noracle: 0.620\nnota: plano e inestable\n")])
f8 = move("attempt", "F8 · decisión **agregadora** + preentrenamiento en dos fases (S3M)",
          under=[q_front], cites=[code(C["s3m"]), config(
              "pipeline: megablend_gru / megablend_cfc\nrun_tag: journal\n"),
              result("oracle: 0.642–0.685\n")])
f9 = move("attempt", "F9 · evidencia **fiel** por coseno congelado",
          under=[q_front], cites=[code(C["ladder"]), config(
              "pipeline: mos_csb_bdimpnet_cos\nrun_tag: csb_cos\nseeds: 5\n"),
              result("cp_f1: 0.379\noracle: 0.453\n")])
f10 = move("attempt", "F10 · evidencia fiel **+ mixer**",
           under=[q_front], cites=[code(C["ladder"]), config(
               "pipeline: mos_csb_bdimpnet_cospost\nrun_tag: csb_cos\nseeds: 5\n"),
               result("cp_f1: 0.513\noracle: 0.631\n")])

f_flat = move("finding",
              "**Ninguna palanca de precisión levanta el readout puro.** Volumen "
              "(F4, 0.628), encoder (F6, 0.620) y hasta el preentrenamiento en dos "
              "fases lo dejan plano en ~0.55–0.62. Sólo la **agregación de canales** "
              "llega a ~0.64 (F8), y a costa de la pureza: la fuga L → 1.",
              under=[f4])
says(f_flat, q_front, "answers", scope=[q_front])
says(f_flat, h0, "refutes", scope=[q_front], in_part=True)

f_cos = move("finding",
             "La evidencia fiel cuesta 15 puntos (F9) **pero el mixer los recupera**: "
             "F10 llega a 0.631, por encima de la base. La única concesión dura casi "
             "se borra.",
             under=[f10])
says(f_cos, q_front, "answers", scope=[q_front], in_part=True)

f_two = move("finding",
             "Reconciliación en dos fases, verificada: **el mismo front-end** "
             "preentrenado da 0.549 con readout lineal puro y 0.642 con decisión "
             "agregadora SNN. La brecha es **la cabeza de decisión**, no los datos. "
             "CfC vs GRU es empate: la célula recurrente no es la palanca.",
             under=[f8])
says(f_two, q_front, "answers", scope=[q_front])

d_front = move("decision",
               "El núcleo del paper deja de ser «tenemos un modelo interpretable» y "
               "pasa a ser **por qué** se colapsa: qué señal es tan fuerte que borra "
               "la interpretabilidad clínica. La frontera interpretabilidad–precisión "
               "es la figura principal.",
               under=[f_flat], scope=[q_front], course="pursue")

# ── Las tres investigaciones bandera ──

q_why = move("question",
             "¿**Dónde** está la señal extra que el modelo puro no puede coger?",
             under=[f_flat])

h_short = move("hypothesis",
               "La ventaja del agregador es el **atajo de auto-revelación** («me "
               "diagnosticaron»), que la descomposición por síntomas resiste. Si al "
               "quitar esos posts la brecha desaparece, la pureza no cuesta nada real.",
               under=[q_why])
a_dis = move("attempt", "Estudio A · ablación de auto-revelación, en los dos modelos",
             under=[h_short], cites=[code(C["disclosure"]), config(
                 "pipelines: indepgru_ft, gru_ft\n"
                 "flags: --drop-disclosure strict / --ablate-disclosure-test\n"
                 "run_tag: csb_disclosure\nWAVE=disclosure ./scripts/enqueue_csb_matrix.sh\n")])
f_dis = move("finding",
             "Sin verificar: los primeros resultados (puro −2.6pp) venían de una "
             "salida a los 2s que era el **bug de argparse al enviar**, no un "
             "resultado. Re-encolado.",
             under=[a_dis])

h_zero = move("hypothesis",
              "Los canales por síntoma entrenados en depresión detectan **otra** "
              "patología sin reentrenar, cambiando sólo las 21 preguntas del "
              "cuestionario. Una caja negra no puede intercambiar «síntomas».",
              under=[q_why])
a_zero = move("attempt", "Estudio B · transferencia cero-shot entre patologías",
              under=[h_zero], cites=[code(C["transfer"]), config(
                  "script: scripts/zero_shot_transfer.py\n"
                  "checkpoints: csb_cos\nqueries: SHI / EAT-26 / PGSI\n")])
a_ext = move("attempt", "Apéndice · CSB interpretable vs GRU-Direct, por patología",
             under=[h_zero], cites=[code(C["ext"]), config(
                 "run_tag: csb_ext\ntasks: selfharm, anorexia, gambling\nseeds: 5\n"),
                 result(
                 "cp_f1/oracle · CSB interpretable vs GRU-Direct\n"
                 "selfharm: 0.536/0.550 vs 0.557/0.580\n"
                 "anorexia: 0.547/0.627 vs 0.630/0.662\n"
                 "gambling: 0.908/0.927 vs 0.949/0.954\n")])
f_ext = move("finding",
             "La interpretabilidad **transfiere** entre patologías —pureza y "
             "causalidad se mantienen— pero va unos puntos por detrás de la caja "
             "negra: −2pp autolesión, −8pp anorexia, −4pp ludopatía. El mismo coste "
             "que enseña la frontera en depresión.",
             under=[a_ext])
says(f_ext, h_zero, "validates", scope=[a_ext], in_part=True)

a_dec = move("attempt", "Estudio C · barrido de decorrelación sobre el agregador",
             under=[q_why], cites=[code(C["grounding"]), config(
                 "pipeline: gru_ft --decorr-weight {0, 0.5, 1, 5}\n"
                 "run_tag: csb_decorr\npregunta: forzar pureza, ¿cae la precisión?\n")])

# ── La grieta: la fidelidad por post ──

h_faith = move("hypothesis",
               "Los canales por síntoma son **fieles a nivel de post**: los posts de "
               "más evidencia para un síntoma hablan de ese síntoma. Es lo que dan a "
               "entender las figuras de caso.",
               under=[q0])

a_audit = move("attempt",
               "Auditoría de fidelidad — ordenar los posts por evidencia y medir el "
               "solape con lo que encuentra un encoder fuerte independiente",
               under=[h_faith], cites=[code(C["signal"]), config(
                   "script: scripts/analysis/faithfulness_audit.py\n"
                   "oráculo: all-mpnet-base-v2\ntarea: anorexia / EAT-26\n"
                   "métrica: faithfulness lift (0 = azar, 1 = el oráculo)\n")])

f_faith = move("finding",
               "**La evidencia por post es casi azar.** Lift +0.07 con la proyección "
               "aprendida (bert-tiny, el modelo del paper) y +0.09 con coseno "
               "congelado, contra 0.301 del oráculo y 0.059 del azar.",
               under=[a_audit])
says(f_faith, h_faith, "refutes", scope=[a_audit])

f_cause = move("finding",
               "**La causa es el encoder, no el enrutado.** Con mpnet-base el lift "
               "sube a +0.31 y con all-mpnet-base-v2 es el oráculo. bert-tiny "
               "(L4/H256) embebe los posts demasiado flojo para que el coseno contra "
               "texto clínico signifique algo.",
               under=[a_audit])
says(f_cause, h_faith, "refutes", scope=[a_audit])

f_picked = move("finding",
                "Las figuras bandera estaban **escogidas a mano**: "
                "`render_fig4_snn_dynamics.py` lleva `PICKED_SPIKES` con los índices "
                "fijados, y no son los de máxima evidencia (Fatiga t=106 era el 3º). "
                "Los posts de máxima evidencia automáticos van fuera de síntoma "
                "incluso en depresión: Culpa → «sex drive from pills».",
                under=[a_audit])
says(f_picked, h_faith, "refutes", scope=[a_audit])

f_stands = move("finding",
                "Lo que **sí** aguanta: `|Δ|_max` mide discriminabilidad entre clases "
                "por canal, y la mide bien. Depresión: SNN .297, CfC .035, clásica "
                ".000. Replica en las extensiones (SHI .295, EAT-26 .407, PGSI .362). "
                "La estabilidad del ranking de síntomas entre semillas es poblacional "
                "y se mantiene.",
                under=[a_audit])
says(f_stands, h0, "validates", scope=[a_audit], in_part=True)

f_bug = move("finding",
             "Bug de reconstrucción de la membrana, ya corregido: las figuras "
             "re-simulaban con θ=1 y β=.9 iniciales en vez de las **aprendidas** por "
             "síntoma, y la membrana divergía a [−15,+6]. La evaluación exporta ahora "
             "la membrana real y θ/β aprendidos; reproduce los spikes al 100%.",
             under=[a_audit])

d_scope = move("decision",
               "**Acotar la afirmación, no retirarla.** Discriminabilidad entre "
               "clases (reclamada, cierta) y fidelidad semántica por post (implícita "
               "en los casos, débil) son propiedades distintas y no se contradicen: "
               "un canal puede separar clases sin que cada post sea léxicamente del "
               "síntoma. Reescribir «enrutado genuino» → «puertas discriminativas "
               "entre clases», etiquetar las figuras de caso como ilustrativas y "
               "seleccionadas, y añadir la auditoría como frontera medida. Ni las "
               "tablas ni la arquitectura cambian.",
               under=[f_cause], scope=[h_faith], course="pursue")

q_granular = move("question",
                  "¿Se puede sacar del modelo discriminativo una figura de caso "
                  "**fiel y granular** —este post disparó este síntoma— que aguante "
                  "una auditoría?",
                  under=[h_faith])

a_regen = move("attempt",
               "Regenerar las figuras de caso con el encoder fuerte congelado. "
               "**No corrido**: queda para si el revisor lo pide.",
               under=[q_granular])

d_curated = move("decision",
                 "Las figuras de caso por usuario y la de dinámica de membrana se "
                 "quedan, etiquetadas como ilustrativas y de selección manual. La "
                 "alternativa —regenerarlas con un encoder fuerte congelado— queda "
                 "aparcada.",
                 under=[f_picked], scope=[a_regen], course="superseded")

# ── El desenlace: dónde muere la fidelidad, y por qué no se pueden las dos ──
#
# Esto es lo que cierra la investigación, y es lo que faltaba: sin ello el
# árbol dejaba `q0` abierta y quien llegaba no podía saber qué salió. Está en
# `docs/INTERPRETABILITY_FINDINGS.md`, bajo «RESOLVED — the tension is
# fundamental».

q_where = move("question",
               "Si la causa es el encoder, ¿con cuál **sí** es fiel la evidencia "
               "por post? Y si con alguno lo es, ¿por qué el modelo entrenado "
               "sigue guardando evidencia ruidosa?",
               under=[f_cause])

a_enc = move("attempt",
             "Cuatro fuentes de evidencia por post, medidas en anorexia",
             under=[q_where], cites=[code(C["rsdd"]), config(
                 "tarea: anorexia / EAT-26\n"
                 "fuentes: bert-tiny proyección aprendida (mos_snn_ft, la del paper)\n"
                 "         bert-tiny coseno congelado\n"
                 "         all-mpnet + DoRA coseno congelado\n"
                 "         all-mpnet totalmente congelado, coseno crudo\n"),
                 result(
                 "fiel por post:\n"
                 "  bert-tiny proyección aprendida: no (≈azar)\n"
                 "  bert-tiny coseno congelado:     no (encoder demasiado flojo)\n"
                 "  all-mpnet + DoRA:               no\n"
                 "  all-mpnet totalmente congelado: SÍ\n")])

f_dora = move("finding",
              "DoRA adapta el encoder de posts mientras el banco de preguntas se "
              "queda en el espacio base: **los dos espacios de embedding se "
              "desincronizan** y el coseno pasa a ser ruido. No es que la "
              "evidencia fiel no exista, es que se la estaba midiendo entre dos "
              "espacios distintos.",
              under=[a_enc])

f_frozen = move("finding",
                "Con all-mpnet **totalmente congelado** —los dos lados en el "
                "espacio base— los posts de más evidencia sí son del síntoma: "
                "purgas, miedo a engordar, contar calorías. La evidencia fiel por "
                "post existe y se puede obtener.",
                under=[a_enc])
says(f_frozen, q_where, "answers", scope=[q_where], in_part=True)

f_norm = move("finding",
              "**Pero el modelo entrenado sigue guardando evidencia ruidosa, aun "
              "con el encoder congelado**, y la raíz está localizada: "
              "`src/models/mos.py:368`, `evidence = self.evidence_norm(evidence)`. "
              "Es un LayerNorm **por post sobre los síntomas**, aplicado al coseno "
              "fiel antes del enrutador: tipifica el vector de síntomas de cada "
              "post (media≈0, std≈1, rango≈[−4,4]). Conserva «qué síntoma es el "
              "más alto **dentro** de este post» y destruye «con cuánta fuerza "
              "este post expresa el síntoma *s* **frente a los demás posts**», que "
              "es exactamente el orden en el que se apoyan las figuras de caso.",
              under=[a_enc])
says(f_norm, q_where, "answers", scope=[q_where])

h_tension = move("hypothesis",
                 "La misma normalización por post que **produce** la "
                 "discriminabilidad poblacional (`|Δ|_max`) es la que **destruye** "
                 "la fidelidad por post. Si es así, no se pueden tener las dos de "
                 "una sola señal, y no es un fallo de implementación: es una "
                 "propiedad de la señal.",
                 under=[f_norm])

a_strong = move("attempt",
                "`mos_snn_cosine_strong_ft` — la variante fiel, a ver si clasifica",
                under=[h_tension], cites=[code(C["rsdd"]), config(
                    "pipeline: mos_snn_cosine_strong_ft\n"
                    "encoder: all-mpnet congelado · backend: python · θ aprendible\n"
                    "seed: 42 · tareas: anorexia (1315), autolesión (1316), "
                    "ludopatía (1317)\n"),
                    result("anorexia · cp_f1: 0.571\nAUROC: 0.90\n|Δ|_max: 0.156 (>.10)\n")])

a_raw = move("attempt",
             "**La prueba decisiva** — simular el LIF sobre el coseno **crudo**, "
             "sin normalizar, y recalcular `|Δ|_max`. Si se mantiene >.10, hay "
             "arreglo: una variante que se salte `evidence_norm` sería fiel y "
             "discriminativa a la vez. Si se hunde, la tensión es fundamental.",
             under=[h_tension], cites=[code(C["rsdd"]), config(
                 "script: scripts/analysis/faithfulness_tension.py\n"
                 "tarea: anorexia · encoder: all-mpnet congelado\n"
                 "umbral de decisión: |Δ|_max > .10 discrimina\n"),
                 result(
                 "señal → |Δ|_max → ¿discrimina?\n"
                 "coseno fiel crudo → puertas LIF : 0.045 · no\n"
                 "media del coseno crudo por usuario: 0.074 · no\n"
                 "evidencia normalizada (el modelo) : 0.156 · sí\n")])

f_tension = move("finding",
                 "**La tensión es fundamental.** La evidencia fiel por post —el "
                 "coseno crudo— **no lleva señal de clase**: 0.045 pasada por el "
                 "LIF, 0.074 promediada por usuario, contra 0.156 de la evidencia "
                 "normalizada. El `evidence_norm` por post no revela la "
                 "discriminabilidad: la **fabrica**, y al fabricarla borra la "
                 "fidelidad. No se pueden tener las dos de esta señal, así que "
                 "entrenar una variante sin normalizar no tiene sentido: no "
                 "discriminaría.",
                 under=[a_raw])
says(f_tension, h_tension, "validates", scope=[a_raw])

f_corrob = move("finding",
                "Corroboración independiente, y ya estaba corrida: las ablaciones "
                "de encoder con **DoRA coseno** dan `|Δ|_max` 0.06–0.16 y cp_F1 "
                "0.42–0.52, es decir, el enrutado por coseno discrimina **peor** "
                "que el bert-tiny aprendido (.297 / .53). «Usar la evidencia "
                "directamente» ya se había probado sin saberlo.",
                under=[a_raw])
says(f_corrob, h_tension, "validates", scope=[a_raw], in_part=True)

f_cohort = move("finding",
                "**Y el lado que sí aguanta.** El mismo coseno fiel crudo eleva "
                "los canales **clínicamente correctos** de cada patología —en "
                "anorexia: evitar hidratos, evitar azúcar, comida de dieta, miedo "
                "a engordar, cortar la comida en trozos pequeños—. Los canales por "
                "síntoma **sí** están fundamentados semánticamente, a nivel de "
                "**cohorte**: para cada trastorno se encienden los síntomas que le "
                "tocan. Lo que no aguanta es el salto de ahí a «este post disparó "
                "este síntoma».",
                under=[a_raw], cites=[code(C["grounded"])])

# El titular cuelga de las dos mitades, y ésa es la forma: no es la conclusión
# negativa ni la positiva, es la composición. Con un solo padre habría que
# elegir cuál de las dos lo produjo, y lo produjeron las dos.
f_headline = move("finding",
                  "**La respuesta a la pregunta de partida: sí a nivel de "
                  "cohorte, no por post, y no de una sola señal.** Se puede "
                  "detectar depresión con una arquitectura interpretable por "
                  "construcción y defender esa interpretabilidad —perfiles por "
                  "síntoma clínicamente coherentes, discriminativos entre clases y "
                  "que transfieren entre patologías—. Lo que no se puede sostener "
                  "es la lectura por post, y no por falta de datos ni de encoder: "
                  "la señal fiel no discrimina y la que discrimina no es fiel.",
                  under=[f_tension, f_cohort])
says(f_headline, q0, "answers", scope=[])
# El titular responde a `q0` y no dice nada de `h0`: lo que le pasa a la
# hipótesis del decodificador puro ya está dicho, y con alcances que no se
# tocan —refutada en rendimiento bajo `q_front`, validada en interpretabilidad
# bajo la auditoría y bajo el coseno crudo—. Colgarle aquí un `refuta` con el
# mismo alcance que el `valida` de al lado la habría dejado en **disputa**, que
# es justo lo que no es: no es que dos personas discrepen, es que la respuesta
# depende de a qué mitad se le pregunte.
says(f_cohort, h0, "validates", scope=[a_raw], in_part=True)

d_final = move("decision",
               "**Liderar con la interpretabilidad poblacional.** Perfiles por "
               "síntoma clínicamente coherentes y discriminativos, `|Δ|_max` y su "
               "estabilidad entre patologías, más la comprobación del coseno crudo "
               "como fundamentación semántica. La auditoría de fidelidad y esta "
               "tensión entran como frontera **medida**, que se adelanta al "
               "revisor y convierte la debilidad en un límite caracterizado. "
               "`|Δ|_max` se reescribe como discriminabilidad entre clases por "
               "canal, no como fidelidad semántica por post.",
               under=[f_headline], scope=[q0], course="pursue")

d_granular = move("decision",
                  "**Retirar la figura de caso fiel y granular como objetivo.** No "
                  "es honestamente obtenible del modelo discriminativo, y ahora se "
                  "sabe por qué. Las figuras de caso que se quedan van etiquetadas "
                  "como ilustrativas y de selección manual.",
                  under=[f_tension], scope=[q_granular], course="abandon")

a_nonorm = move("attempt",
                "Entrenar una variante que se salte `evidence_norm` en la evidencia "
                "externa: fiel **y** discriminativa. **No corrido.**",
                under=[h_tension])

d_nonorm = move("decision",
                "No entrenar la variante sin `evidence_norm`. Era el arreglo obvio "
                "y la prueba decisiva dice que no discriminaría: la señal fiel no "
                "lleva clase, así que no hay nada que la normalización esté "
                "tapando.",
                under=[f_tension], scope=[a_nonorm], course="abandon")

# El alcance de una decisión de abandono apunta a **lo que se abandona**, y por
# eso lo abandonado tiene que ser un movimiento. Apuntarlo a la hipótesis que lo
# produjo —lo único que hay si el intento no se escribe— plegaba la línea
# entera, titular incluido: el árbol se quedaba sin el hallazgo que responde a
# la pregunta de partida. Un intento que nadie corrió sigue siendo un
# movimiento, y es justo el que hace falta para poder decir que no se corrió.

print(json.dumps({"q0": q0, "movimientos": "sembrados"}))
