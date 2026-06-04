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

let cache = { loadedAt: 0, files: {}, records: [] };
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
  return /\b(healthcare|hospital|clinical|patient|doctor|nurse|care|medical|health|command center)\b/i.test(query);
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

    records.push({
      sourceFile: fileName,
      title: item.title || item.name || item.concept || item.id || fallbackTitle || fileName,
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

  return cache;
}

function score(record, query) {
  const qTokens = tokens(query);
  const phrase = normalize(query);
  const haystack = normalize(`${record.title} ${record.text} ${record.sourceFile}`);
  const title = normalize(record.title);

  let value = 0;

  if (title === phrase) value += 120;
  if (title.includes(phrase)) value += 60;
  if (phrase.includes(title) && title.length > 2) value += 40;

  qTokens.forEach(token => {
    if (title === token) value += 50;
    if (title.includes(token)) value += 18;
    if (haystack.includes(token)) value += 3;
  });

  if (record.sourceFile === "ask-cortex-answers.json") value += 25;
  if (record.sourceFile === "governance-impact-map.json") value += 18;
  if (record.sourceFile === "relationship-graph.json") value += 8;
  if (record.sourceFile === "cortex-items.json") value += 6;
  if (record.sourceFile === "knowledge-graph-v2.json") value += 5;
  if (record.sourceFile === "search-index-v2.json") value -= 5;

  if (!isHealthcareQuery(query) && /healthcare|hospital|clinical|patient|medical|health/i.test(haystack)) {
    value -= 60;
  }

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
  const answers = cortexData.files["ask-cortex-answers.json"];

  if (!Array.isArray(answers)) return [];

  return answers
    .filter(item => {
      const concept = normalize(item.concept || item.title || item.name);
      return qTokens.some(token => concept === token) || normalize(query).includes(concept);
    })
    .map(item => ({
      record: {
        sourceFile: "ask-cortex-answers.json",
        title: item.concept || item.title || item.name,
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
  const impactMap = cortexData.files["governance-impact-map.json"] || {};
  const matches = [];

  Object.entries(impactMap).forEach(([key, value]) => {
    const keyNorm = normalize(key);
    const isMatch =
      qTokens.some(token => keyNorm === token) ||
      normalize(query).includes(keyNorm);

    if (isMatch) {
      matches.push({
        subject: key,
        values: value,
        impactScore: 999
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

  const topRecords = dedupeRecords([...exactAskMatches, ...scored]).slice(0, 10);

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

        if (selected.has(source)) relScore += 60;
        if (selected.has(target)) relScore += 60;

        qTokens.forEach(token => {
          if (source === token) relScore += 80;
          if (target === token) relScore += 80;
          if (source.includes(token)) relScore += 20;
          if (target.includes(token)) relScore += 20;
        });

        if (!isHealthcareQuery(query) && /healthcare|hospital|clinical|patient|medical|health/i.test(`${rel.source} ${rel.target}`)) {
          relScore -= 80;
        }

        return { rel, relScore };
      })
      .filter(item => item.relScore > 0)
      .sort((a, b) => b.relScore - a.relScore)
      .slice(0, 8)
      .map(item => item.rel)
    : [];

  const impactMap = cortexData.files["governance-impact-map.json"] || {};
  const broadImpacts = [];

  Object.entries(impactMap).forEach(([key, value]) => {
    const keyNorm = normalize(key);
    let impactScore = 0;

    if (selected.has(keyNorm)) impactScore += 80;

    qTokens.forEach(token => {
      if (keyNorm === token) impactScore += 100;
      if (keyNorm.includes(token)) impactScore += 25;
    });

    if (!isHealthcareQuery(query) && /healthcare|hospital|clinical|patient|medical|health/i.test(key)) {
      impactScore -= 80;
    }

    if (impactScore > 0) {
      broadImpacts.push({ subject: key, values: value, impactScore });
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
    .slice(0, 6);

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

function cleanConceptTitle(title) {
  return String(title || "").trim();
}

function buildGuaranteedSourceTrace(retrieved) {
  const conceptSeen = new Set();
  const concepts = [];

  retrieved.topRecords.forEach(x => {
    const title = cleanConceptTitle(x.record.title);
    const key = normalize(title);

    if (!key || conceptSeen.has(key)) return;
    conceptSeen.add(key);
    concepts.push(`- ${title}`);
  });

  const relationships = retrieved.relationships
    .slice(0, 5)
    .map(rel => `- ${rel.source} → ${rel.type || "related"} → ${rel.target}`)
    .join("\n") || "- No direct relationships found";

  const governance = retrieved.impacts
    .slice(0, 5)
    .map(impact => {
      const values = Array.isArray(impact.values)
        ? impact.values.join(", ")
        : String(impact.values);
      return `- ${impact.subject}: ${values}`;
    })
    .join("\n") || "- No governance impacts found";

  const sources = [
    ...new Set(retrieved.topRecords.map(x => x.record.sourceFile))
  ]
    .slice(0, 6)
    .map(source => `- ${source}`)
    .join("\n") || "- No source files found";

  const confidence = Math.min(95, 65 + (retrieved.topRecords.length * 3));

  return `

Source Trace
------------
Concepts:
${concepts.slice(0, 5).join("\n") || "- No direct concepts found"}

Relationships:
${relationships}

Governance:
${governance}

Sources:
${sources}

Confidence:
${confidence}%`;
}

function removeModelSourceTrace(reply) {
  return String(reply || "")
    .replace(/source trace[\s\S]*$/i, "")
    .trim();
}

const systemPrompt = `
You are Cortex AI, the source-grounded assistant for NABDs.AI Cortex 10.

Use retrieved Cortex knowledge when relevant.

Rules:
- Start with a direct answer.
- Be clear, executive-friendly, practical, and concise.
- Prefer Cortex source knowledge over generic AI knowledge.
- Mention relationship trace when relationships are provided.
- Mention governance implications when governance impacts are provided.
- Include practical next steps when useful.
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
            `Answer using the retrieved Cortex knowledge. Do not include a Source Trace section.`
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
