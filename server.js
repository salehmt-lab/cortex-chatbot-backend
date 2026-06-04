import express from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";
import OpenAI from "openai";

const app = express();
const port = process.env.PORT || 10000;

app.use(cors({
  origin: ["https://nabds.ai", "https://www.nabds.ai"]
}));

app.use(express.json({ limit: "1mb" }));

app.use("/chat", rateLimit({
  windowMs: 60 * 1000,
  max: 20
}));

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const model = process.env.OPENAI_MODEL || "gpt-4.1-mini";

const DATA_BASE = "https://nabds.ai/cortex/data/";

const DATA_FILES = [
  "ask-cortex-answers.json",
  "relationship-graph.json",
  "governance-impact-map.json",
  "cortex-items.json",
  "knowledge-graph-v2.json",
  "search-index-v2.json"
];

const BLOCKED_TRACE_CONCEPTS = new Set([
  "contact",
  "contacts",
  "strategy",
  "home",
  "about",
  "overview",
  "cortex dictionary",
  "101 ai terms explained",
  "multi agent systems",
  "multi-agent systems",
  "site navigation",
  "navigation",
  "footer",
  "hero",
  "call to action",
"cta",
"ai terms explained",
  "knowledge articles",
  "knowledge hub",
  "article learning paths",
  "operations overview",]);

const BLOCKED_GOVERNANCE_SUBJECTS = new Set([
  "contact",
  "contacts",
  "strategy",
  "home",
  "about",
  "overview",
  "cortex dictionary",
  "101 ai terms explained",
  "multi agent systems",
  "multi-agent systems",
  "site navigation",
  "navigation",
  "footer",
  "hero",
  "call to action",
  "cta"

  "ai terms explained",
  "knowledge articles",
  "knowledge hub",
  "article learning paths",
  "operations overview",]);

const ALLOWED_TRACE_SOURCES = new Set([
  "ask-cortex-answers.json",
  "governance-impact-map.json",
  "relationship-graph.json"
]);

let cache = { loadedAt: 0, files: {}, records: [], errors: [] };
const CACHE_TTL = 10 * 60 * 1000;

function normalize(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function tokens(value) {
  const text = normalize(value);
  return text ? text.split(/\s+/).filter(t => t.length > 1) : [];
}

function isHealthcareQuery(query) {
  return /\b(healthcare|hospital|clinical|patient|doctor|nurse|care|medical|health|command center|icu|er|surgery|diagnosis|treatment)\b/i.test(query);
}

function isGovernanceQuery(query) {
  return /\b(governance|control|controls|risk|compliance|audit|lineage|traceability|approval|policy|policies|guardrail|guardrails|impact|impacts|privacy|access)\b/i.test(query);
}

function isRelationshipQuery(query) {
  return /\b(related|relationship|relationships|connect|connected|dependency|dependencies|depends|link|linked|graph|trace)\b/i.test(query);
}

function isDefinitionQuery(query) {
  return /\b(what is|define|explain|overview|describe|meaning|definition)\b/i.test(query);
}

function recordText(record) {
  return [
    record.id,
    record.title,
    record.name,
    record.concept,
    record.type,
    record.domain,
    record.summary,
    record.answer,
    record.businessImpact,
    record.text,
    Array.isArray(record.relatedConcepts) ? record.relatedConcepts.join(" ") : "",
    Array.isArray(record.relatedObjects) ? record.relatedObjects.join(" ") : "",
    Array.isArray(record.dependencies) ? record.dependencies.join(" ") : "",
    Array.isArray(record.governanceImpacts) ? record.governanceImpacts.join(" ") : "",
    Array.isArray(record.governanceNotes) ? record.governanceNotes.join(" ") : "",
    record.sections ? JSON.stringify(record.sections).slice(0, 1200) : ""
  ].filter(Boolean).join(" ");
}

function flatten(fileName, data) {
  const records = [];

  function add(item, fallbackTitle) {
    if (!item || typeof item !== "object") return;

    const title = item.title || item.name || item.concept || item.id || fallbackTitle || fileName;

    records.push({
      sourceFile: fileName,
      title,
      normalizedTitle: normalize(title),
      type: item.type || item.group || "knowledge_object",
      domain: item.domain || item.group || "",
      raw: item,
      text: recordText(item)
    });
  }

  if (Array.isArray(data)) {
    data.forEach((item, index) => add(item, `${fileName} item ${index + 1}`));
  } else if (data && typeof data === "object") {
    if (Array.isArray(data.nodes)) {
      data.nodes.forEach((item, index) => add(item, `node ${index + 1}`));
    }

    if (Array.isArray(data.edges)) {
      data.edges.forEach((item, index) => add(item, `edge ${index + 1}`));
    }

    Object.entries(data).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        add({
          title: key,
          summary: value.join(", "),
          type: "mapped_values"
        }, key);
      } else if (value && typeof value === "object") {
        add(value, key);
      }
    });
  }

  return records;
}

async function loadCortexData() {
  const now = Date.now();

  if (cache.records.length && now - cache.loadedAt < CACHE_TTL) {
    return cache;
  }

  const files = {};
  const records = [];
  const errors = [];

  for (const fileName of DATA_FILES) {
    try {
      const response = await fetch(`${DATA_BASE}${fileName}?v=${Date.now()}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      files[fileName] = data;
      records.push(...flatten(fileName, data));
    } catch (error) {
      errors.push(`${fileName}: ${error.message}`);
    }
  }

  cache = { loadedAt: now, files, records, errors };
  console.log(`Loaded Cortex RAG data: ${Object.keys(files).length} files, ${records.length} records`);

  if (errors.length) {
    console.warn("Cortex data loading warnings:", errors.join(" | "));
  }

  return cache;
}

function isBlockedTraceTitle(title) {
  return BLOCKED_TRACE_CONCEPTS.has(normalize(title));
}

function isBlockedGovernanceSubject(subject) {
  return BLOCKED_GOVERNANCE_SUBJECTS.has(normalize(subject));
}

function isHealthcareRecord(record) {
  const text = normalize(`${record.title} ${record.text} ${record.domain}`);
  return /\b(healthcare|hospital|clinical|patient|medical|health)\b/i.test(text);
}

function score(record, query) {
  const qTokens = tokens(query);
  const phrase = normalize(query);
  const haystack = normalize(`${record.title} ${record.text} ${record.sourceFile}`);
  const title = normalize(record.title);

  let value = 0;

  if (isBlockedTraceTitle(record.title)) value -= 80;

  if (title === phrase) value += 140;
  if (title.includes(phrase) && phrase.length > 2) value += 65;
  if (phrase.includes(title) && title.length > 2) value += 45;

  qTokens.forEach(token => {
    if (title === token) value += 60;
    if (title.includes(token)) value += 20;
    if (haystack.includes(token)) value += 3;
  });

  if (record.sourceFile === "ask-cortex-answers.json") value += 30;
  if (record.sourceFile === "governance-impact-map.json") value += isGovernanceQuery(query) ? 22 : 12;
  if (record.sourceFile === "relationship-graph.json") value += isRelationshipQuery(query) ? 20 : 8;
  if (record.sourceFile === "cortex-items.json") value += 4;
  if (record.sourceFile === "knowledge-graph-v2.json") value += 4;
  if (record.sourceFile === "search-index-v2.json") value -= 10;

  if (isDefinitionQuery(query) && record.sourceFile === "ask-cortex-answers.json") value += 15;
  if (!isHealthcareQuery(query) && isHealthcareRecord(record)) value -= 75;

  return value;
}

function dedupeRecords(items) {
  const seen = new Set();
  const result = [];

  for (const item of items) {
    const key = normalize(item.record.title);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    result.push(item);
  }

  return result;
}

function findExactAskCortexMatches(query, cortexData) {
  const qTokens = tokens(query);
  const phrase = normalize(query);
  const answers = cortexData.files["ask-cortex-answers.json"];

  if (!Array.isArray(answers)) return [];

  return answers
    .filter(item => {
      const concept = normalize(item.concept || item.title || item.name);
      if (!concept || isBlockedTraceTitle(concept)) return false;

      return qTokens.some(token => concept === token) ||
        phrase.includes(concept) ||
        concept.includes(phrase);
    })
    .map(item => ({
      record: {
        sourceFile: "ask-cortex-answers.json",
        title: item.concept || item.title || item.name,
        normalizedTitle: normalize(item.concept || item.title || item.name),
        type: "concept_answer",
        domain: "Cortex",
        raw: item,
        text: recordText(item)
      },
      score: 999
    }));
}

function findExactGovernanceMatches(query, cortexData) {
  const qTokens = tokens(query);
  const phrase = normalize(query);
  const impactMap = cortexData.files["governance-impact-map.json"] || {};
  const matches = [];

  Object.entries(impactMap).forEach(([key, value]) => {
    const keyNorm = normalize(key);
    if (!keyNorm || isBlockedGovernanceSubject(key)) return;

    const isMatch =
      qTokens.some(token => keyNorm === token) ||
      phrase.includes(keyNorm) ||
      keyNorm.includes(phrase);

    if (isMatch) {
      matches.push({
        subject: key,
        values: value,
        impactScore: 999,
        exact: true
      });
    }
  });

  return matches;
}

function retrieve(query, cortexData) {
  const qTokens = tokens(query);

  const exactAskMatches = findExactAskCortexMatches(query, cortexData);
  const exactGovernanceMatches = findExactGovernanceMatches(query, cortexData);

  const scored = cortexData.records
    .map(record => ({ record, score: score(record, query) }))
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score);

  const topRecords = dedupeRecords([...exactAskMatches, ...scored])
    .filter(item => !isBlockedTraceTitle(item.record.title))
    .slice(0, 10);

  const selected = new Set();

  topRecords.forEach(({ record }) => {
    selected.add(normalize(record.title));
    if (record.raw?.concept) selected.add(normalize(record.raw.concept));
    if (record.raw?.name) selected.add(normalize(record.raw.name));
  });

  qTokens.forEach(token => selected.add(token));

  const relationships = Array.isArray(cortexData.files["relationship-graph.json"])
    ? cortexData.files["relationship-graph.json"]
      .map(rel => {
        const source = normalize(rel.source);
        const target = normalize(rel.target);
        let relScore = 0;

        if (selected.has(source)) relScore += 70;
        if (selected.has(target)) relScore += 70;

        qTokens.forEach(token => {
          if (source === token) relScore += 90;
          if (target === token) relScore += 90;
          if (source.includes(token)) relScore += 22;
          if (target.includes(token)) relScore += 22;
        });

        if (!isHealthcareQuery(query) && /healthcare|hospital|clinical|patient|medical|health/i.test(`${rel.source} ${rel.target}`)) {
          relScore -= 100;
        }

        return { rel, relScore };
      })
      .filter(item => item.relScore > 0)
      .sort((a, b) => b.relScore - a.relScore)
      .slice(0, isRelationshipQuery(query) ? 8 : 5)
      .map(item => item.rel)
    : [];

  const impactMap = cortexData.files["governance-impact-map.json"] || {};
  const broadImpacts = [];

  Object.entries(impactMap).forEach(([key, value]) => {
    const keyNorm = normalize(key);
    if (!keyNorm || isBlockedGovernanceSubject(key)) return;

    let impactScore = 0;

    if (selected.has(keyNorm)) impactScore += 85;

    qTokens.forEach(token => {
      if (keyNorm === token) impactScore += 110;
      if (keyNorm.includes(token)) impactScore += 28;
    });

    if (!isHealthcareQuery(query) && /healthcare|hospital|clinical|patient|medical|health/i.test(key)) {
      impactScore -= 100;
    }

    if (impactScore > 0) {
      broadImpacts.push({ subject: key, values: value, impactScore, exact: false });
    }
  });

  const impactSeen = new Set();
  const impacts = [...exactGovernanceMatches, ...broadImpacts]
    .sort((a, b) => b.impactScore - a.impactScore)
    .filter(item => {
      const key = normalize(item.subject);
      if (impactSeen.has(key)) return false;
      impactSeen.add(key);
      return true;
    })
    .slice(0, isGovernanceQuery(query) ? 4 : 3);

  return { topRecords, relationships, impacts };
}

function buildContext(retrieved) {
  const lines = ["RETRIEVED CORTEX KNOWLEDGE:"];

  retrieved.topRecords.forEach(({ record, score }, index) => {
    const raw = record.raw || {};
    const content = [
      raw.summary,
      raw.answer,
      raw.businessImpact,
      raw.text,
      raw.governanceNotes ? `Governance notes: ${raw.governanceNotes.join(", ")}` : "",
      raw.relatedObjects ? `Related objects: ${raw.relatedObjects.join(", ")}` : "",
      raw.relatedConcepts ? `Related concepts: ${raw.relatedConcepts.join(", ")}` : "",
      raw.sections ? JSON.stringify(raw.sections).slice(0, 900) : ""
    ].filter(Boolean).join("\n");

    lines.push(
      `[${index + 1}] ${record.title}\n` +
      `Source: ${record.sourceFile}\n` +
      `Type: ${record.type}\n` +
      `Domain: ${record.domain || "Cortex"}\n` +
      `Score: ${score}\n` +
      `Content: ${content.slice(0, 1200)}`
    );
  });

  if (retrieved.relationships.length) {
    lines.push("\nRELATED CORTEX RELATIONSHIPS:");
    retrieved.relationships.forEach((rel, index) => {
      lines.push(`[R${index + 1}] ${rel.source} -> ${rel.type || "related"} -> ${rel.target}`);
    });
  }

  if (retrieved.impacts.length) {
    lines.push("\nGOVERNANCE IMPACTS:");
    retrieved.impacts.forEach((impact, index) => {
      const values = Array.isArray(impact.values) ? impact.values.join(", ") : String(impact.values);
      lines.push(`[G${index + 1}] ${impact.subject}: ${values}`);
    });
  }

  return lines.join("\n\n").slice(0, 14000);
}

function calculateConfidence(retrieved) {
  const hasExactConcept = retrieved.topRecords.some(item => item.score >= 999);
  const hasGovernance = retrieved.impacts.length > 0;
  const hasRelationship = retrieved.relationships.length > 0;
  const hasSources = retrieved.topRecords.length > 0;

  if (hasExactConcept && hasGovernance && hasRelationship) return 95;
  if (hasExactConcept && (hasGovernance || hasRelationship)) return 90;
  if (hasExactConcept) return 85;
  if (hasSources && (hasGovernance || hasRelationship)) return 78;
  if (hasSources) return 70;
  return 50;
}

function buildGuaranteedSourceTrace(retrieved) {
  const conceptSeen = new Set();
  const concepts = [];

  .sort((a, b) =>
    (b.record.raw?.tracePriority || 0) -
    (a.record.raw?.tracePriority || 0)
  )
  .forEach(x => {
    const title = String(x.record.title || "").trim();
    const key = normalize(title);

    if (!key || conceptSeen.has(key) || isBlockedTraceTitle(title)) return;

    // Only authoritative Cortex RAG assets should appear as trace concepts.
    // This removes broad page/article objects such as "AI Terms Explained".
   if (!ALLOWED_TRACE_SOURCES.has(x.record.sourceFile)) return;
if (x.record.raw?.allowTrace === false) return;

conceptSeen.add(key);
concepts.push(`- ${title}`);
  });

  const relationshipSeen = new Set();
  const relationships = [];

  retrieved.relationships.forEach(rel => {
    const source = String(rel.source || "").trim();
    const target = String(rel.target || "").trim();
    const type = String(rel.type || "related").trim();
    const key = normalize(`${source} ${type} ${target}`);

    if (!source || !target || relationshipSeen.has(key)) return;

    relationshipSeen.add(key);
    relationships.push(`- ${source} → ${type} → ${target}`);
  });

  const governanceSeen = new Set();
  const governanceItems = [];

  retrieved.impacts
    .filter(impact => !isBlockedGovernanceSubject(impact.subject))
    .filter(impact => impact.exact || impact.impactScore >= 80)
    .forEach(impact => {
      const subject = String(impact.subject || "").trim();
      const key = normalize(subject);

      if (!subject || governanceSeen.has(key)) return;

      governanceSeen.add(key);

      const values = Array.isArray(impact.values)
        ? impact.values.join(", ")
        : String(impact.values);

      governanceItems.push(`- ${subject}: ${values}`);
    });

  const sourceSeen = new Set();
  const sources = [];

  retrieved.topRecords.forEach(x => {
    const source = x.record.sourceFile;

    if (!source || sourceSeen.has(source)) return;
    if (!ALLOWED_TRACE_SOURCES.has(source)) return;

    sourceSeen.add(source);
    sources.push(`- ${source}`);
  });

  const confidence = calculateConfidence(retrieved);

  return `

Source Trace
------------
Concepts:
${concepts.slice(0, 5).join("\n") || "- No direct concepts found"}

Relationships:
${relationships.slice(0, 4).join("\n") || "- No direct relationships found"}

Governance:
${governanceItems.slice(0, 3).join("\n") || "- No governance impacts found"}

Sources:
${sources.slice(0, 6).join("\n") || "- No source files found"}

Confidence:
${confidence}%`;
}
function removeModelSourceTrace(reply) {
  return String(reply || "")
    .replace(/\nSource Trace[\s\S]*$/i, "")
    .trim();
}

const systemPrompt = `
You are Cortex AI, the source-grounded assistant for NABDs.AI Cortex 10.

Use retrieved Cortex knowledge when relevant.

Rules:
- Start with a direct answer.
- Keep the answer complete and concise: usually 1 to 2 short paragraphs.
- Prefer Cortex source knowledge over generic AI knowledge.
- Mention relationship trace only when relationships are provided and relevant.
- Mention governance implications only when governance impacts are provided and relevant.
- Include practical next steps only when useful.
- Do not invent pricing, customers, certifications, commitments, legal claims, or medical claims.
- If the retrieved Cortex context does not support the answer, say: "I do not see that in the current Cortex knowledge assets."
- Do not mention backend, Render, OpenAI, GoDaddy, or implementation details to normal website visitors.
- Do not create your own Source Trace section. The backend will append it.
`;

app.get("/", (req, res) => {
  res.send("Cortex RAG chatbot backend is running.");
});

app.get("/health", async (req, res) => {
  try {
    const cortexData = await loadCortexData();

    res.json({
      status: "ok",
      message: "Cortex RAG chatbot backend is running.",
      loadedFiles: Object.keys(cortexData.files).length,
      records: cortexData.records.length,
      cacheLoadedAt: new Date(cortexData.loadedAt).toISOString(),
      errors: cortexData.errors || []
    });
  } catch (error) {
    res.status(500).json({ status: "error", error: error.message });
  }
});

app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body?.message;

    if (!userMessage || typeof userMessage !== "string") {
      return res.status(400).json({ error: "Message is required." });
    }

    const cortexData = await loadCortexData();
    const retrieved = retrieve(userMessage, cortexData);
    const context = buildContext(retrieved);

    const response = await client.responses.create({
      model,
      input: [
        { role: "system", content: systemPrompt },
        {
          role: "user",
          content:
            `User Question:\n${userMessage}\n\n` +
            `${context}\n\n` +
            `Answer using the retrieved Cortex knowledge. Keep the answer complete, concise, and do not include a Source Trace section.`
        }
      ]
    });

    const cleanReply = removeModelSourceTrace(
      response.output_text || "I am sorry, I could not generate a response."
    );

    res.json({
      reply: cleanReply + buildGuaranteedSourceTrace(retrieved),
      sources: retrieved.topRecords.slice(0, 6).map(({ record, score }) => ({
        title: record.title,
        sourceFile: record.sourceFile,
        type: record.type,
        domain: record.domain,
        score
      })),
      relationships: retrieved.relationships.slice(0, 6),
      governanceImpacts: retrieved.impacts.slice(0, 6)
    });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ error: "Cortex RAG chatbot backend error." });
  }
});

app.listen(port, () => {
  console.log(`Cortex RAG chatbot backend running on port ${port}`);
});
