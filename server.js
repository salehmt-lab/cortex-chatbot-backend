import express from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";
import OpenAI from "openai";

const app = express();
const port = process.env.PORT || 10000;

const allowedOrigins = [
  "https://nabds.ai",
  "https://www.nabds.ai"
];

app.use(cors({
  origin: allowedOrigins
}));

app.use(express.json({ limit: "1mb" }));

const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 20
});

app.use("/chat", limiter);

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const model = process.env.OPENAI_MODEL || "gpt-4.1-mini";

const cortexSystemPrompt = `
You are Cortex AI, the assistant for NABDs.AI Cortex 10.

Cortex 10 is an internal enterprise intelligence operating system that connects knowledge, governance, relationships, learning, executive intelligence, and operational decision support.

You help users understand Cortex concepts, governance, architecture, agentic AI, operations, healthcare AI, academy content, and source-traced relationships.

Core positioning:
- Cortex 10 is an Enterprise Intelligence Operating System.
- It includes structured intelligence layers such as knowledge graph, dictionary, Cortex AI assistant, trust model, executive intelligence, role views, learning objects, semantic search, contribution workflow, and maturity model.
- It supports domains including context architecture, agentic AI, and AI governance.
- It is designed for internal use and governed enterprise decision support.

Answer style:
- Be clear, executive-friendly, and practical.
- Explain Cortex in business language first, then technical language if useful.
- Keep answers concise unless the user asks for detail.
- Help users navigate Cortex capabilities and concepts.
- When useful, suggest next steps such as exploring governance, maturity models, relationship graphs, learning objects, or executive intelligence.

Boundaries:
- Do not invent pricing.
- Do not claim external certifications.
- Do not invent customers, clients, or partnerships.
- Do not provide legal, medical, or compliance guarantees.
- Do not expose internal-only assumptions as facts.
- If unsure, say the information is not available in the current Cortex knowledge base.

Lead/contact behavior:
If someone asks for access, partnership, demo, or implementation support, guide them to request access or contact the NABDs.AI team through the Cortex site.
`;

app.get("/", (req, res) => {
  res.send("Cortex chatbot backend is running.");
});

app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body?.message;

    if (!userMessage || typeof userMessage !== "string") {
      return res.status(400).json({ error: "Message is required." });
    }

    const response = await client.responses.create({
      model,
      input: [
        {
          role: "system",
          content: cortexSystemPrompt
        },
        {
          role: "user",
          content: userMessage
        }
      ]
    });

    res.json({
      reply: response.output_text || "I am sorry, I could not generate a response."
    });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({
      error: "Cortex chatbot backend error."
    });
  }
});

app.listen(port, () => {
  console.log(`Cortex chatbot backend running on port ${port}`);
});
