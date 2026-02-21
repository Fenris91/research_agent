import { useState, useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";

// ═══════════════════════════════════════════════════════════════
// KNOWLEDGE EXPLORER — Split-Pane Layout
//
//   ┌──────────────────┬─────────────────────────────┐
//   │   CONVERSATION   │      SEMANTIC GRAPH         │
//   │   with agent     │      force-directed map     │
//   │                  │                             │
//   │   chat history   │   calm physics              │
//   │   + input        │   click = expand in place   │
//   └──────────────────┴─────────────────────────────┘
// ═══════════════════════════════════════════════════════════════

const RESEARCHERS = [
  {
    id: "r1", label: "David Harvey", institution: "CUNY",
    hIndex: 98, color: "#c9953e", fields: ["urban geography", "political economy", "marxism"],
    papers: [
      { id: "p1", label: "The Right to the City", year: 2008, citations: 4200, field: "urban geography", oa: "green",
        doi: "10.1080/13604810802164953", tldr: "Examines the right to the city as a collective right to reshape urbanization." },
      { id: "p2", label: "Neoliberalism as Creative Destruction", year: 2007, citations: 3100, field: "political economy", oa: "amber",
        doi: "10.1111/j.1467-8330.2007.00529.x", tldr: "Analyzes how neoliberalism became hegemonic as a mode of discourse." },
      { id: "p3", label: "Spaces of Capital", year: 2001, citations: 2800, field: "urban geography", oa: "red",
        doi: "10.4324/9780203975954", tldr: "Critical geography exploring the production of space under capitalism." },
    ]
  },
  {
    id: "r2", label: "Doreen Massey", institution: "Open University",
    hIndex: 74, color: "#8878c0", fields: ["space and place", "power geometry", "globalization"],
    papers: [
      { id: "p4", label: "A Global Sense of Place", year: 1991, citations: 5100, field: "space and place", oa: "green",
        doi: "10.1007/978-1-349-24951-4_29", tldr: "Proposes a progressive sense of place defined by social relations beyond boundaries." },
      { id: "p5", label: "Space, Place and Gender", year: 1994, citations: 4600, field: "space and place", oa: "amber",
        doi: "10.5749/j.ctttsg8p", tldr: "Explores how space, place, and gender are deeply interconnected." },
      { id: "p6", label: "Power-geometries", year: 1993, citations: 2200, field: "power geometry", oa: "green",
        doi: "10.1007/978-1-349-24951-4_28", tldr: "Different social groups have distinct relationships to mobility and flows." },
    ]
  },
  {
    id: "r3", label: "Anna Tsing", institution: "UC Santa Cruz",
    hIndex: 52, color: "#5098ab", fields: ["multispecies ethnography", "supply chains", "anthropocene"],
    papers: [
      { id: "p7", label: "The Mushroom at the End of the World", year: 2015, citations: 3800, field: "multispecies ethnography", oa: "red",
        doi: "10.1515/9781400873548", tldr: "Explores the matsutake mushroom to examine precarity and capitalism." },
      { id: "p8", label: "Friction", year: 2005, citations: 5200, field: "supply chains", oa: "green",
        doi: "10.1515/9781400830596", tldr: "Studies how global connections are made through friction across difference." },
      { id: "p9", label: "Supply Chains and the Human Condition", year: 2009, citations: 1200, field: "supply chains", oa: "amber",
        doi: "10.1080/09502380903424124", tldr: "Examines how supply chain capitalism transforms labor and nature." },
    ]
  },
];

const CROSS_CITATIONS = [
  { source: "p4", target: "p1", intent: "method" },
  { source: "p3", target: "p6", intent: "background" },
  { source: "p8", target: "p4", intent: "background" },
  { source: "p7", target: "p2", intent: "result" },
  { source: "p5", target: "p3", intent: "method" },
  { source: "p1", target: "p5", intent: "background" },
];

function semanticScore(query, text) {
  const q = query.toLowerCase(), t = text.toLowerCase();
  const words = q.split(/\s+/).filter(w => w.length > 3);
  if (!words.length) return 0;
  let hits = 0;
  for (const w of words) { if (t.includes(w)) hits++; else { const s = w.slice(0, Math.max(4, w.length - 2)); if (t.includes(s)) hits += 0.4; } }
  return Math.min(1, hits / words.length);
}

const OA = { green: { color: "#22c55e", label: "Open Access" }, amber: { color: "#eab308", label: "Preprint" }, red: { color: "#64748b", label: "Paywalled" } };
const INTENT_C = { background: "#64748b", method: "#3b82f6", result: "#22c55e" };

const CHAT_HISTORY = [
  { role: "user", text: "Can you help me explore literature on urban displacement and gentrification?" },
  { role: "agent", text: "I'll search across OpenAlex and Semantic Scholar for papers on urban displacement and gentrification. Let me build a citation network around the key works.\n\nI found 3 key researchers and 9 foundational papers. The graph is loading on the right — David Harvey's work on the right to the city is central, with strong citation links to Doreen Massey's spatial theory.\n\nTry clicking on any node to see details, or search for specific concepts to see how they connect semantically." },
  { role: "user", text: "How does Anna Tsing's work connect to urban geography?" },
  { role: "agent", text: "Interesting question. Tsing's supply chain ethnography doesn't cite the urban geography literature directly, but there are semantic bridges — her work on \"friction\" and global connections shares conceptual territory with Massey's \"power-geometries.\" Harvey's neoliberalism work also intersects with Tsing's critique of capitalism.\n\nYou can see these connections in the graph as dashed semantic edges. Try searching \"capitalism space\" to highlight the overlap." },
];

// ═══════════════════════════════════════════════════════════════

export default function KnowledgeExplorerSplit() {
  const svgRef = useRef(null);
  const graphContainerRef = useRef(null);
  const chatEndRef = useRef(null);
  const simRef = useRef(null);
  const nodesRef = useRef(null);
  const linksRef = useRef(null);

  const [dims, setDims] = useState({ w: 600, h: 500 });
  const [selected, setSelected] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [activeQuery, setActiveQuery] = useState(null);
  const [searchInput, setSearchInput] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState(CHAT_HISTORY);
  const [settled, setSettled] = useState(false);

  // Build graph data
  const buildGraph = useCallback(() => {
    const nodes = [], links = [], nodeSet = new Set();
    RESEARCHERS.forEach(r => {
      nodes.push({ id: r.id, type: "researcher", label: r.label, sub: `h-index: ${r.hIndex} · ${r.institution}`, color: r.color, size: 24, fields: r.fields, hIndex: r.hIndex, institution: r.institution });
      nodeSet.add(r.id);
      r.papers.forEach(p => {
        const sz = 7 + Math.min(12, Math.sqrt(p.citations / 80));
        nodes.push({ id: p.id, type: "paper", label: p.label, sub: `${p.year} · ${p.citations} cites`, color: r.color, size: sz, year: p.year, citations: p.citations, field: p.field, oa: p.oa, doi: p.doi, tldr: p.tldr, researcherId: r.id, researcherLabel: r.label });
        nodeSet.add(p.id);
        links.push({ source: r.id, target: p.id, type: "authorship", color: r.color, strength: 0.6 });
      });
    });
    CROSS_CITATIONS.forEach(c => {
      if (nodeSet.has(c.source) && nodeSet.has(c.target))
        links.push({ source: c.source, target: c.target, type: "citation", intent: c.intent, color: INTENT_C[c.intent], strength: 0.08 });
    });
    if (activeQuery?.trim()) {
      const qId = "query";
      nodes.push({ id: qId, type: "query", label: `"${activeQuery}"`, sub: "search query", color: "#c45c4a", size: 18 });
      RESEARCHERS.flatMap(r => r.papers.map(p => ({ ...p, researcher: r }))).forEach(p => {
        const s = Math.max(semanticScore(activeQuery, p.label) * 0.4, semanticScore(activeQuery, p.field) * 0.35, semanticScore(activeQuery, p.tldr || "") * 0.35)
          + Math.max(...p.researcher.fields.map(f => semanticScore(activeQuery, f))) * 0.25;
        if (s > 0.15) links.push({ source: qId, target: p.id, type: "semantic", color: "#c45c4a", strength: s * 0.4, score: s });
      });
      RESEARCHERS.forEach(r => {
        const fs = Math.max(...r.fields.map(f => semanticScore(activeQuery, f)));
        if (fs > 0.2) links.push({ source: qId, target: r.id, type: "semantic", color: "#c45c4a", strength: fs * 0.2, score: fs });
      });
    }
    return { nodes, links };
  }, [activeQuery]);

  // Get connected node IDs
  const getConnected = useCallback((nodeId, links) => {
    const ids = new Set([nodeId]);
    links.forEach(l => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      if (s === nodeId) ids.add(t);
      if (t === nodeId) ids.add(s);
    });
    return ids;
  }, []);

  // D3 render
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    const { w, h } = dims;
    const data = buildGraph();
    setSettled(false);

    const defs = svg.append("defs");
    data.nodes.forEach(n => {
      const gr = defs.append("radialGradient").attr("id", `n-${n.id}`).attr("cx", "35%").attr("cy", "35%");
      gr.append("stop").attr("offset", "0%").attr("stop-color", n.color).attr("stop-opacity", 0.28);
      gr.append("stop").attr("offset", "100%").attr("stop-color", n.color).attr("stop-opacity", 0.04);
    });
    const glow = defs.append("filter").attr("id", "gw").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
    glow.append("feGaussianBlur").attr("stdDeviation", "4").attr("result", "b");
    const gm = glow.append("feMerge"); gm.append("feMergeNode").attr("in", "b"); gm.append("feMergeNode").attr("in", "SourceGraphic");

    const root = svg.append("g");
    svg.call(d3.zoom().scaleExtent([0.35, 2.5]).on("zoom", e => root.attr("transform", e.transform)));

    // Ambient halos
    root.append("g").selectAll("circle").data(data.nodes.filter(n => n.type !== "paper")).join("circle")
      .attr("r", d => d.size * 2.2).attr("fill", d => d.color).attr("opacity", 0.015);

    // Links
    const linkG = root.append("g");
    const linkEls = linkG.selectAll("line").data(data.links).join("line")
      .attr("stroke", d => d.color)
      .attr("stroke-width", d => d.type === "semantic" ? 0.8 + (d.score || 0) * 2 : d.type === "citation" ? 0.8 : 1)
      .attr("stroke-dasharray", d => d.type === "citation" ? "3,3" : d.type === "semantic" ? "5,3" : "none")
      .attr("opacity", d => d.type === "semantic" ? 0.1 + (d.score || 0) * 0.35 : d.type === "citation" ? 0.1 : 0.12);

    const scoreLabels = linkG.selectAll("text")
      .data(data.links.filter(l => l.type === "semantic" && (l.score || 0) > 0.3))
      .join("text").text(d => `${Math.round(d.score * 100)}%`)
      .attr("font-size", "7px").attr("fill", "#c45c4a").attr("opacity", 0)
      .attr("text-anchor", "middle").attr("font-family", "'IBM Plex Mono', monospace")
      .style("pointer-events", "none");

    // Nodes
    const nodeG = root.append("g");
    const nodeEls = nodeG.selectAll("g").data(data.nodes).join("g").style("cursor", "pointer");

    // OA rings
    nodeEls.filter(d => d.type === "paper").append("circle")
      .attr("class", "oa-ring")
      .attr("r", d => d.size + 2.5)
      .attr("fill", "none")
      .attr("stroke", d => OA[d.oa]?.color || "#475569")
      .attr("stroke-width", 1.5)
      .attr("opacity", 0.5)
      .attr("stroke-dasharray", d => d.oa === "red" ? "2,2" : "none");

    // Researcher outer ring
    nodeEls.filter(d => d.type === "researcher").append("circle")
      .attr("class", "outer-ring")
      .attr("r", d => d.size + 3)
      .attr("fill", "none").attr("stroke", d => d.color)
      .attr("stroke-width", 0.4).attr("opacity", 0.15);

    // Main circle
    nodeEls.append("circle").attr("class", "body")
      .attr("r", d => d.size)
      .attr("fill", d => `url(#n-${d.id})`)
      .attr("stroke", d => d.type === "query" ? "#c45c4a" : d.color)
      .attr("stroke-width", d => d.type === "researcher" ? 1.5 : d.type === "query" ? 1.2 : 0.8)
      .attr("stroke-dasharray", d => d.type === "query" ? "3,2" : "none")
      .attr("opacity", 0.9)
      .attr("filter", d => (d.type === "researcher" || d.type === "query") ? "url(#gw)" : null);

    // Icons
    nodeEls.filter(d => d.type === "researcher").append("text")
      .text("◎").attr("text-anchor", "middle").attr("dy", 4)
      .attr("font-size", "12px").attr("fill", d => d.color).attr("opacity", 0.3).style("pointer-events", "none");
    nodeEls.filter(d => d.type === "query").append("text")
      .text("⊙").attr("text-anchor", "middle").attr("dy", 4)
      .attr("font-size", "10px").attr("fill", "#c45c4a").attr("opacity", 0.35).style("pointer-events", "none");

    // Labels
    nodeEls.append("text").attr("class", "label")
      .text(d => d.label.length > 24 ? d.label.slice(0, 22) + "…" : d.label)
      .attr("text-anchor", "middle").attr("dy", d => -d.size - 6)
      .attr("font-size", d => d.type === "researcher" ? "9px" : d.type === "query" ? "8px" : "6.5px")
      .attr("font-weight", d => d.type === "researcher" || d.type === "query" ? 600 : 400)
      .attr("fill", d => d.type === "query" ? "#c45c4a" : d.type === "researcher" ? d.color : "#6a7590")
      .attr("letter-spacing", d => d.type === "researcher" ? "0.8px" : "0.2px")
      .attr("font-family", "'IBM Plex Mono', monospace")
      .style("pointer-events", "none");

    // Sub-labels (hidden by default, shown on select)
    nodeEls.append("text").attr("class", "sub-label")
      .text(d => d.sub)
      .attr("text-anchor", "middle").attr("dy", d => d.size + 12)
      .attr("font-size", "6px").attr("fill", "#4a5568").attr("opacity", 0)
      .attr("font-family", "'IBM Plex Mono', monospace")
      .style("pointer-events", "none");

    // Field tag (hidden by default)
    nodeEls.filter(d => d.type === "paper").append("text").attr("class", "field-tag")
      .text(d => d.field || "")
      .attr("text-anchor", "middle").attr("dy", d => d.size + 22)
      .attr("font-size", "5.5px").attr("fill", d => d.color).attr("opacity", 0)
      .attr("font-family", "'IBM Plex Mono', monospace")
      .style("pointer-events", "none");

    nodesRef.current = nodeEls;
    linksRef.current = linkEls;

    // === CALM HOVER: no physics changes, just visual emphasis ===
    nodeEls.on("mouseenter", (ev, d) => {
      if (selected) return; // don't override selection
      setHovered(d);
      const conn = getConnected(d.id, data.links);

      nodeEls.transition().duration(250).ease(d3.easeCubicOut)
        .attr("opacity", n => conn.has(n.id) ? 1 : 0.15);
      nodeEls.selectAll(".body").transition().duration(250).ease(d3.easeCubicOut)
        .attr("r", n => conn.has(n.id) && n.id === d.id ? n.size + 3 : n.size);
      nodeEls.selectAll(".sub-label").transition().duration(250)
        .attr("opacity", n => n.id === d.id ? 0.7 : 0);
      nodeEls.selectAll(".field-tag").transition().duration(250)
        .attr("opacity", n => conn.has(n.id) ? 0.5 : 0);

      linkEls.transition().duration(250).ease(d3.easeCubicOut)
        .attr("opacity", l => {
          const s = typeof l.source === "object" ? l.source.id : l.source;
          const t = typeof l.target === "object" ? l.target.id : l.target;
          return (s === d.id || t === d.id) ? 0.6 : 0.03;
        })
        .attr("stroke-width", l => {
          const s = typeof l.source === "object" ? l.source.id : l.source;
          const t = typeof l.target === "object" ? l.target.id : l.target;
          const isConn = s === d.id || t === d.id;
          if (!isConn) return l.type === "semantic" ? 0.8 + (l.score || 0) * 2 : l.type === "citation" ? 0.8 : 1;
          return l.type === "semantic" ? 1.5 + (l.score || 0) * 3 : l.type === "citation" ? 2 : 2;
        });

      scoreLabels.transition().duration(250)
        .attr("opacity", l => {
          const s = typeof l.source === "object" ? l.source.id : l.source;
          const t = typeof l.target === "object" ? l.target.id : l.target;
          return (s === d.id || t === d.id) ? 0.7 : 0;
        });
    })
    .on("mouseleave", (ev, d) => {
      if (selected) return;
      setHovered(null);
      resetVisuals(nodeEls, linkEls, scoreLabels, data);
    })
    .on("click", (ev, d) => {
      ev.stopPropagation();
      setSelected(prev => {
        const next = prev?.id === d.id ? null : d;
        if (next) {
          applySelection(next, nodeEls, linkEls, scoreLabels, data, getConnected);
        } else {
          resetVisuals(nodeEls, linkEls, scoreLabels, data);
        }
        return next;
      });
    });

    // Click background to deselect
    svg.on("click", () => {
      setSelected(null);
      setHovered(null);
      resetVisuals(nodeEls, linkEls, scoreLabels, data);
    });

    // Drag — gentle, no alpha spike
    nodeEls.call(d3.drag()
      .on("start", (ev, d) => { d.fx = d.x; d.fy = d.y; })
      .on("drag", (ev, d) => {
        d.fx = ev.x; d.fy = ev.y;
        if (simRef.current) { simRef.current.alphaTarget(0.02).restart(); }
      })
      .on("end", (ev, d) => {
        d.fx = null; d.fy = null;
        if (simRef.current) { simRef.current.alphaTarget(0); }
      })
    );

    // Force simulation — settles quickly, low energy
    const sim = d3.forceSimulation(data.nodes)
      .force("link", d3.forceLink(data.links).id(d => d.id)
        .distance(d => d.type === "authorship" ? 55 + d.target.size * 2 : d.type === "semantic" ? 100 : 140)
        .strength(d => d.strength || 0.2))
      .force("charge", d3.forceManyBody().strength(d => d.type === "researcher" ? -280 : d.type === "query" ? -200 : -50))
      .force("center", d3.forceCenter(w / 2, h / 2).strength(0.03))
      .force("collision", d3.forceCollide().radius(d => d.size + 8).strength(0.7))
      .alpha(0.8)
      .alphaDecay(0.04) // settles faster
      .velocityDecay(0.55) // more friction = calmer
      .on("tick", () => {
        linkEls.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        scoreLabels.attr("x", d => (d.source.x + d.target.x) / 2).attr("y", d => (d.source.y + d.target.y) / 2 - 4);
        nodeEls.attr("transform", d => `translate(${d.x},${d.y})`);
        root.select("g:first-child").selectAll("circle")
          .data(data.nodes.filter(n => n.type !== "paper"))
          .attr("cx", d => d.x).attr("cy", d => d.y);
      })
      .on("end", () => setSettled(true));

    simRef.current = sim;
    return () => sim.stop();
  }, [dims, buildGraph, getConnected, selected]);

  // Resize
  useEffect(() => {
    const el = graphContainerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(entries => {
      for (const e of entries) { const { width, height } = e.contentRect; if (width > 0) setDims({ w: width, h: height }); }
    });
    obs.observe(el);
    setDims({ w: el.offsetWidth || 600, h: el.offsetHeight || 500 });
    return () => obs.disconnect();
  }, []);

  // Scroll chat to bottom
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chatMessages]);

  const handleSearch = () => {
    if (searchInput.trim()) { setActiveQuery(searchInput.trim()); setSelected(null); }
  };
  const handleClearSearch = () => { setActiveQuery(null); setSearchInput(""); setSelected(null); };

  const handleChatSend = () => {
    if (!chatInput.trim()) return;
    setChatMessages(prev => [...prev, { role: "user", text: chatInput.trim() }]);
    const q = chatInput.trim();
    setChatInput("");
    setTimeout(() => {
      setChatMessages(prev => [...prev, { role: "agent", text: `Searching for "${q}" across OpenAlex and Semantic Scholar... The graph will update with relevant papers and connections.` }]);
      setActiveQuery(q);
    }, 800);
  };

  const displayNode = selected || hovered;

  // ═══ THEME ═══
  const font = "'IBM Plex Mono', 'Fira Code', monospace";
  const bg = "#0a0d13";
  const surface = "#0e1219";
  const border = "#1a1f2e";
  const t1 = "#c8d0e0";
  const t2 = "#5a6580";
  const t3 = "#2e3648";

  return (
    <div style={{ width: "100%", height: "100vh", display: "flex", background: bg, fontFamily: font, color: t1, overflow: "hidden" }}>

      {/* ════════ LEFT: CONVERSATION ════════ */}
      <div style={{
        width: "38%", minWidth: 300, maxWidth: 460, height: "100%",
        background: surface, borderRight: `1px solid ${border}`,
        display: "flex", flexDirection: "column"
      }}>
        {/* Header */}
        <div style={{ padding: "14px 18px 12px", borderBottom: `1px solid ${border}`, flexShrink: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#c45c4a" }} />
            <span style={{ fontSize: "11px", fontWeight: 600, letterSpacing: "1.5px", color: t1 }}>RESEARCH AGENT</span>
          </div>
          <div style={{ fontSize: "8px", color: t3, letterSpacing: "0.8px", marginTop: 3 }}>
            KNOWLEDGE EXPLORER · 3 RESEARCHERS · 9 PAPERS
          </div>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflow: "auto", padding: "12px 18px", display: "flex", flexDirection: "column", gap: 12 }}>
          {chatMessages.map((msg, i) => (
            <div key={i} style={{
              alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "88%"
            }}>
              {msg.role === "agent" && (
                <div style={{ fontSize: "7px", color: t3, letterSpacing: "0.8px", marginBottom: 3, fontWeight: 600 }}>AGENT</div>
              )}
              <div style={{
                background: msg.role === "user" ? "#1a2236" : "#111620",
                border: `1px solid ${msg.role === "user" ? "#252e45" : border}`,
                borderRadius: msg.role === "user" ? "12px 12px 4px 12px" : "12px 12px 12px 4px",
                padding: "10px 13px", fontSize: "11px", lineHeight: 1.6,
                color: msg.role === "user" ? "#d0d8ea" : "#9aa3b8",
                whiteSpace: "pre-wrap"
              }}>
                {msg.text}
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <div style={{ padding: "12px 18px", borderTop: `1px solid ${border}`, flexShrink: 0 }}>
          <div style={{ display: "flex", gap: 6 }}>
            <input value={chatInput} onChange={e => setChatInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleChatSend()}
              placeholder="Ask the research agent..."
              style={{
                flex: 1, background: "#0a0e16", border: `1px solid ${border}`, borderRadius: 8,
                padding: "10px 12px", color: t1, fontSize: "11px", fontFamily: font, outline: "none"
              }} />
            <button onClick={handleChatSend} style={{
              background: "#c45c4a15", border: `1px solid #c45c4a30`, borderRadius: 8,
              padding: "10px 14px", cursor: "pointer", color: "#c45c4a",
              fontSize: "9px", fontWeight: 600, fontFamily: font, letterSpacing: "0.5px"
            }}>SEND</button>
          </div>
        </div>
      </div>

      {/* ════════ RIGHT: GRAPH ════════ */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Graph toolbar */}
        <div style={{
          padding: "8px 14px", borderBottom: `1px solid ${border}`, flexShrink: 0,
          display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", background: surface
        }}>
          <div style={{ display: "flex", gap: 0, flex: 1, maxWidth: 320, minWidth: 180,
            background: bg, border: `1px solid ${border}`, borderRadius: 6 }}>
            <input value={searchInput} onChange={e => setSearchInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSearch()}
              placeholder="Semantic search..."
              style={{
                flex: 1, background: "transparent", border: "none", outline: "none",
                color: t1, padding: "6px 10px", fontSize: "10px", fontFamily: font
              }} />
            <button onClick={handleSearch} style={{
              background: "transparent", border: "none", borderLeft: `1px solid ${border}`,
              color: "#c45c4a", padding: "6px 10px", cursor: "pointer",
              fontSize: "8px", fontWeight: 600, fontFamily: font, letterSpacing: "0.5px"
            }}>SEARCH</button>
          </div>

          {activeQuery && (
            <button onClick={handleClearSearch} style={{
              background: "transparent", border: `1px solid ${border}`, borderRadius: 4,
              color: t2, padding: "4px 8px", cursor: "pointer", fontSize: "8px", fontFamily: font
            }}>✕ CLEAR</button>
          )}

          {/* Legends inline */}
          <div style={{ marginLeft: "auto", display: "flex", gap: 10, alignItems: "center" }}>
            {[
              { c: t2, l: "AUTHOR", d: false },
              { c: "#475569", l: "CITES", d: true },
              { c: "#c45c4a", l: "SEMANTIC", d: true },
            ].map(x => (
              <div key={x.l} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                <svg width="14" height="2"><line x1="0" y1="1" x2="14" y2="1" stroke={x.c} strokeWidth="1.5" strokeDasharray={x.d ? "3,2" : "none"} opacity="0.6" /></svg>
                <span style={{ fontSize: "6px", color: t3, letterSpacing: "0.8px", fontWeight: 600 }}>{x.l}</span>
              </div>
            ))}
            <div style={{ width: 1, height: 10, background: border }} />
            {Object.entries(OA).map(([k, v]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 2 }}>
                <div style={{ width: 5, height: 5, borderRadius: "50%", border: `1.5px solid ${v.color}`, background: `${v.color}15` }} />
                <span style={{ fontSize: "5.5px", color: t3, letterSpacing: "0.6px", fontWeight: 600 }}>{v.label.toUpperCase()}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Graph + detail panel */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden", position: "relative" }}>

          {/* SVG canvas */}
          <div ref={graphContainerRef} style={{
            flex: 1, background: `radial-gradient(ellipse at 50% 40%, #0e141f 0%, ${bg} 65%)`,
            position: "relative"
          }}>
            <svg ref={svgRef} width={dims.w} height={dims.h} style={{ display: "block" }} />

            {/* Suggestion pills */}
            {!activeQuery && (
              <div style={{
                position: "absolute", bottom: 14, left: "50%", transform: "translateX(-50%)",
                display: "flex", gap: 5, zIndex: 5
              }}>
                <span style={{ fontSize: "7px", color: t3, alignSelf: "center" }}>TRY:</span>
                {["urban space capitalism", "supply chains global", "place gender power"].map(q => (
                  <button key={q} onClick={() => { setSearchInput(q); setActiveQuery(q); }} style={{
                    background: `${bg}cc`, border: `1px solid ${border}`, borderRadius: 10,
                    padding: "3px 9px", cursor: "pointer", color: t2,
                    fontSize: "7px", fontFamily: font, letterSpacing: "0.3px",
                    backdropFilter: "blur(8px)", transition: "all 0.2s"
                  }}
                  onMouseEnter={e => { e.target.style.borderColor = "#c45c4a40"; e.target.style.color = "#c45c4a"; }}
                  onMouseLeave={e => { e.target.style.borderColor = border; e.target.style.color = t2; }}
                  >{q}</button>
                ))}
              </div>
            )}
          </div>

          {/* Detail panel — slides in from right */}
          {displayNode && (
            <div style={{
              width: 230, minWidth: 230, background: surface,
              borderLeft: `1px solid ${border}`, overflow: "auto",
              display: "flex", flexDirection: "column",
              animation: "none"
            }}>
              {/* Header */}
              <div style={{ padding: "12px 14px 10px", borderBottom: `1px solid ${border}` }}>
                <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 5 }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: displayNode.color, opacity: 0.7 }} />
                  <span style={{ fontSize: "7px", fontWeight: 600, color: displayNode.color, letterSpacing: "1px" }}>
                    {displayNode.type.toUpperCase()}
                  </span>
                  {displayNode.oa && (
                    <span style={{
                      marginLeft: "auto", fontSize: "6px", fontWeight: 600,
                      color: OA[displayNode.oa].color,
                      background: `${OA[displayNode.oa].color}10`,
                      border: `1px solid ${OA[displayNode.oa].color}25`,
                      borderRadius: 6, padding: "1px 5px", letterSpacing: "0.3px"
                    }}>{OA[displayNode.oa].label.toUpperCase()}</span>
                  )}
                </div>
                <div style={{ fontSize: "12px", fontWeight: 600, color: t1, lineHeight: 1.35 }}>
                  {displayNode.label}
                </div>
                <div style={{ fontSize: "8px", color: t2, marginTop: 3 }}>{displayNode.sub}</div>
              </div>

              <div style={{ padding: "10px 14px", flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
                {displayNode.type === "paper" && (
                  <>
                    {displayNode.field && (
                      <span style={{
                        alignSelf: "flex-start", background: `${displayNode.color}0c`,
                        border: `1px solid ${displayNode.color}20`, borderRadius: 6,
                        padding: "2px 7px", fontSize: "7px", color: displayNode.color
                      }}>{displayNode.field}</span>
                    )}
                    {displayNode.tldr && (
                      <div>
                        <div style={{ fontSize: "7px", color: t3, letterSpacing: "0.8px", fontWeight: 600, marginBottom: 2 }}>TLDR</div>
                        <div style={{ fontSize: "9px", color: t2, lineHeight: 1.5 }}>{displayNode.tldr}</div>
                      </div>
                    )}
                    {displayNode.doi && (
                      <div>
                        <div style={{ fontSize: "7px", color: t3, letterSpacing: "0.8px", fontWeight: 600, marginBottom: 2 }}>DOI</div>
                        <div style={{ fontSize: "7px", color: "#5a7eaa", wordBreak: "break-all" }}>{displayNode.doi}</div>
                      </div>
                    )}
                    <div style={{ display: "flex", flexDirection: "column", gap: 3, marginTop: 4 }}>
                      {displayNode.oa === "green" && <Btn color="#22c55e" label="↓ DOWNLOAD PDF" />}
                      {displayNode.oa === "amber" && <Btn color="#eab308" label="↓ GET PREPRINT" />}
                      {displayNode.oa === "red" && (
                        <div style={{ fontSize: "7px", color: t2, textAlign: "center", padding: "5px", background: "#ffffff03", borderRadius: 4, border: `1px solid ${border}` }}>
                          🔒 Abstract + metadata only
                        </div>
                      )}
                      <Btn color={t2} label="EXPAND CITATIONS →" />
                      <Btn color={t2} label="FIND SIMILAR PAPERS" />
                    </div>
                  </>
                )}

                {displayNode.type === "researcher" && (
                  <>
                    <div style={{ display: "flex", gap: 14 }}>
                      <div>
                        <div style={{ fontSize: "7px", color: t3, letterSpacing: "0.8px", fontWeight: 600, marginBottom: 2 }}>H-INDEX</div>
                        <div style={{ fontSize: "18px", fontWeight: 700, color: displayNode.color }}>{displayNode.hIndex}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: "7px", color: t3, letterSpacing: "0.8px", fontWeight: 600, marginBottom: 2 }}>INSTITUTION</div>
                        <div style={{ fontSize: "9px", color: t1, marginTop: 2 }}>{displayNode.institution}</div>
                      </div>
                    </div>
                    {displayNode.fields && (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                        {displayNode.fields.map(f => (
                          <span key={f} style={{
                            background: `${displayNode.color}0c`, border: `1px solid ${displayNode.color}18`,
                            borderRadius: 5, padding: "2px 6px", fontSize: "6.5px", color: displayNode.color
                          }}>{f}</span>
                        ))}
                      </div>
                    )}
                    <div style={{ display: "flex", flexDirection: "column", gap: 3, marginTop: 4 }}>
                      <Btn color={displayNode.color} label="LOAD ALL PAPERS" />
                      <Btn color={t2} label="CO-AUTHOR NETWORK" />
                      <Btn color={t2} label="VIEW PROFILE →" />
                    </div>
                  </>
                )}

                {displayNode.type === "query" && (
                  <div style={{ fontSize: "8px", color: t2, lineHeight: 1.5 }}>
                    Dashed edges show semantic similarity from SPECTER2 embeddings. Percentage = cosine similarity score.
                  </div>
                )}
              </div>

              {selected && (
                <div style={{ padding: "8px 14px", borderTop: `1px solid ${border}`, flexShrink: 0 }}>
                  <button onClick={() => { setSelected(null); resetVisuals(nodesRef.current, linksRef.current, null, buildGraph()); }}
                    style={{ width: "100%", background: "transparent", border: `1px solid ${border}`, borderRadius: 4, padding: "5px", cursor: "pointer", color: t2, fontSize: "7px", fontFamily: font, letterSpacing: "0.5px" }}>
                    DESELECT
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ═══ Helper: Small button ═══
function Btn({ color, label }) {
  const [h, setH] = useState(false);
  return (
    <button
      onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}
      style={{
        background: h ? `${color}12` : "transparent",
        border: `1px solid ${h ? color + "35" : "#1a1f2e"}`,
        borderRadius: 4, padding: "5px 8px", cursor: "pointer",
        color: h ? color : "#5a6580", fontSize: "7px", fontWeight: 600,
        fontFamily: "'IBM Plex Mono', monospace", letterSpacing: "0.5px",
        textAlign: "center", width: "100%", transition: "all 0.2s"
      }}>{label}</button>
  );
}

// ═══ Visual helpers: no simulation changes ═══
function resetVisuals(nodeEls, linkEls, scoreLabels, data) {
  if (!nodeEls || !linkEls) return;
  nodeEls.transition().duration(350).ease(d3.easeCubicOut).attr("opacity", 1);
  nodeEls.selectAll(".body").transition().duration(350).ease(d3.easeCubicOut)
    .attr("r", d => d.size);
  nodeEls.selectAll(".sub-label").transition().duration(250).attr("opacity", 0);
  nodeEls.selectAll(".field-tag").transition().duration(250).attr("opacity", 0);
  linkEls.transition().duration(350).ease(d3.easeCubicOut)
    .attr("opacity", d => d.type === "semantic" ? 0.1 + (d.score || 0) * 0.35 : d.type === "citation" ? 0.1 : 0.12)
    .attr("stroke-width", d => d.type === "semantic" ? 0.8 + (d.score || 0) * 2 : d.type === "citation" ? 0.8 : 1);
  if (scoreLabels) scoreLabels.transition().duration(250).attr("opacity", 0);
}

function applySelection(node, nodeEls, linkEls, scoreLabels, data, getConnected) {
  if (!nodeEls || !linkEls) return;
  const conn = getConnected(node.id, data.links);

  nodeEls.transition().duration(300).ease(d3.easeCubicOut)
    .attr("opacity", n => conn.has(n.id) ? 1 : 0.1);
  nodeEls.selectAll(".body").transition().duration(300).ease(d3.easeCubicOut)
    .attr("r", n => n.id === node.id ? n.size + 4 : n.size);
  nodeEls.selectAll(".sub-label").transition().duration(300)
    .attr("opacity", n => conn.has(n.id) ? 0.7 : 0);
  nodeEls.selectAll(".field-tag").transition().duration(300)
    .attr("opacity", n => conn.has(n.id) ? 0.55 : 0);

  linkEls.transition().duration(300).ease(d3.easeCubicOut)
    .attr("opacity", l => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      return (s === node.id || t === node.id) ? 0.65 : 0.02;
    })
    .attr("stroke-width", l => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      if (s !== node.id && t !== node.id) return l.type === "citation" ? 0.8 : 1;
      return l.type === "semantic" ? 2 + (l.score || 0) * 3 : l.type === "citation" ? 2.5 : 2.5;
    });

  if (scoreLabels) scoreLabels.transition().duration(300)
    .attr("opacity", l => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      return (s === node.id || t === node.id) ? 0.8 : 0;
    });
}
