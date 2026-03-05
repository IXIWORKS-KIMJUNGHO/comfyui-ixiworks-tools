import { app } from "../../scripts/app.js";

function getSelectedNodes(canvas) {
    return Object.values(canvas.selected_nodes || {});
}

function alignNodes(nodes, direction) {
    if (nodes.length < 2) return;
    switch (direction) {
        case "left": {
            const minX = Math.min(...nodes.map((n) => n.pos[0]));
            nodes.forEach((n) => (n.pos[0] = minX));
            break;
        }
        case "right": {
            const maxX = Math.max(...nodes.map((n) => n.pos[0] + n.size[0]));
            nodes.forEach((n) => (n.pos[0] = maxX - n.size[0]));
            break;
        }
        case "top": {
            const minY = Math.min(...nodes.map((n) => n.pos[1]));
            nodes.forEach((n) => (n.pos[1] = minY));
            break;
        }
        case "bottom": {
            const maxY = Math.max(...nodes.map((n) => n.pos[1] + n.size[1]));
            nodes.forEach((n) => (n.pos[1] = maxY - n.size[1]));
            break;
        }
        case "centerH": {
            const centerX = nodes.reduce((s, n) => s + n.pos[0] + n.size[0] / 2, 0) / nodes.length;
            nodes.forEach((n) => (n.pos[0] = centerX - n.size[0] / 2));
            break;
        }
        case "centerV": {
            const centerY = nodes.reduce((s, n) => s + n.pos[1] + n.size[1] / 2, 0) / nodes.length;
            nodes.forEach((n) => (n.pos[1] = centerY - n.size[1] / 2));
            break;
        }
    }
    app.graph.setDirtyCanvas(true, true);
}

function matchSize(nodes, mode) {
    if (nodes.length < 2) return;
    switch (mode) {
        case "width": {
            const maxW = Math.max(...nodes.map((n) => n.size[0]));
            nodes.forEach((n) => (n.size[0] = maxW));
            break;
        }
        case "height": {
            const maxH = Math.max(...nodes.map((n) => n.size[1]));
            nodes.forEach((n) => (n.size[1] = maxH));
            break;
        }
        case "both": {
            const maxW = Math.max(...nodes.map((n) => n.size[0]));
            const maxH = Math.max(...nodes.map((n) => n.size[1]));
            nodes.forEach((n) => { n.size[0] = maxW; n.size[1] = maxH; });
            break;
        }
    }
    app.graph.setDirtyCanvas(true, true);
}

const GAP = 30;

function distributeNodes(nodes, axis) {
    if (nodes.length < 2) return;
    if (axis === "horizontal") {
        nodes.sort((a, b) => a.pos[0] - b.pos[0]);
        let x = nodes[0].pos[0] + nodes[0].size[0] + GAP;
        for (let i = 1; i < nodes.length; i++) {
            nodes[i].pos[0] = x;
            x += nodes[i].size[0] + GAP;
        }
    } else {
        nodes.sort((a, b) => a.pos[1] - b.pos[1]);
        let y = nodes[0].pos[1] + nodes[0].size[1] + GAP;
        for (let i = 1; i < nodes.length; i++) {
            nodes[i].pos[1] = y;
            y += nodes[i].size[1] + GAP;
        }
    }
    app.graph.setDirtyCanvas(true, true);
}

// Chord shortcuts: hold Z/X/C/V + press number key
const CHORD_MAP = {
    z: { 1: "align:left", 2: "align:right", 3: "align:top", 4: "align:bottom" },
    x: { 1: "center:centerH", 2: "center:centerV" },
    c: { 1: "dist:horizontal", 2: "dist:vertical" },
    v: { 1: "match:width", 2: "match:height", 3: "match:both" },
};

function executeChord(action) {
    const selected = getSelectedNodes(app.canvas);
    const [group, param] = action.split(":");
    if (group === "align" || group === "center") alignNodes(selected, param);
    else if (group === "dist") distributeNodes(selected, param);
    else if (group === "match") matchSize(selected, param);
}

// Floating toolbar
function createToolbar() {
    const style = document.createElement("style");
    style.textContent = `
        #ixi-align-toolbar {
            position: fixed;
            top: 40px;
            left: 50%;
            transform: translateX(-50%);
            background: #1e1e2e;
            border: 1px solid #555;
            border-radius: 8px;
            padding: 4px 8px;
            display: none;
            align-items: center;
            gap: 2px;
            z-index: 9999;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
            font-family: system-ui, sans-serif;
            font-size: 11px;
        }
        #ixi-align-toolbar .sep {
            width: 1px;
            height: 20px;
            background: #555;
            margin: 0 4px;
        }
        #ixi-align-toolbar button {
            background: transparent;
            border: none;
            color: #ccc;
            cursor: pointer;
            padding: 4px 6px;
            border-radius: 4px;
            font-size: 11px;
            line-height: 1;
            white-space: nowrap;
        }
        #ixi-align-toolbar button:hover {
            background: #333;
            color: #fff;
        }
        #ixi-align-toolbar .grp-label {
            color: #888;
            font-size: 10px;
            padding: 0 2px;
            user-select: none;
        }
    `;
    document.head.appendChild(style);

    const bar = document.createElement("div");
    bar.id = "ixi-align-toolbar";

    const groups = [
        {
            label: "Align",
            items: [
                { text: "L", title: "Left (Z+1)", fn: () => alignNodes(getSelectedNodes(app.canvas), "left") },
                { text: "R", title: "Right (Z+2)", fn: () => alignNodes(getSelectedNodes(app.canvas), "right") },
                { text: "T", title: "Top (Z+3)", fn: () => alignNodes(getSelectedNodes(app.canvas), "top") },
                { text: "B", title: "Bottom (Z+4)", fn: () => alignNodes(getSelectedNodes(app.canvas), "bottom") },
            ]
        },
        {
            label: "Center",
            items: [
                { text: "H", title: "Horizontal (X+1)", fn: () => alignNodes(getSelectedNodes(app.canvas), "centerH") },
                { text: "V", title: "Vertical (X+2)", fn: () => alignNodes(getSelectedNodes(app.canvas), "centerV") },
            ]
        },
        {
            label: "Dist",
            items: [
                { text: "H", title: "Horizontal (C+1)", fn: () => distributeNodes(getSelectedNodes(app.canvas), "horizontal") },
                { text: "V", title: "Vertical (C+2)", fn: () => distributeNodes(getSelectedNodes(app.canvas), "vertical") },
            ]
        },
        {
            label: "Size",
            items: [
                { text: "W", title: "Width (V+1)", fn: () => matchSize(getSelectedNodes(app.canvas), "width") },
                { text: "H", title: "Height (V+2)", fn: () => matchSize(getSelectedNodes(app.canvas), "height") },
                { text: "WH", title: "Both (V+3)", fn: () => matchSize(getSelectedNodes(app.canvas), "both") },
            ]
        },
    ];

    groups.forEach((group, gi) => {
        if (gi > 0) {
            const sep = document.createElement("div");
            sep.className = "sep";
            bar.appendChild(sep);
        }
        const label = document.createElement("span");
        label.className = "grp-label";
        label.textContent = group.label;
        bar.appendChild(label);
        group.items.forEach((item) => {
            const btn = document.createElement("button");
            btn.textContent = item.text;
            btn.title = item.title;
            btn.addEventListener("click", item.fn);
            bar.appendChild(btn);
        });
    });

    document.body.appendChild(bar);
    return bar;
}

app.registerExtension({
    name: "IXIWORKS.NodeAlign",

    async setup() {
        // Floating toolbar - show when 2+ nodes selected
        const toolbar = createToolbar();
        let lastCount = 0;
        setInterval(() => {
            const canvas = app.canvas;
            if (!canvas) return;
            const count = getSelectedNodes(canvas).length;
            if (count >= 2 && lastCount < 2) {
                toolbar.style.display = "flex";
            } else if (count < 2 && lastCount >= 2) {
                toolbar.style.display = "none";
            }
            lastCount = count;
        }, 200);

        // Chord shortcuts
        let heldKey = null;

        window.addEventListener("keydown", (e) => {
            if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.isContentEditable) return;
            const key = e.key.toLowerCase();

            if (key in CHORD_MAP && !heldKey) {
                heldKey = key;
                e.preventDefault();
                e.stopPropagation();
                return;
            }

            if (heldKey && CHORD_MAP[heldKey]) {
                const action = CHORD_MAP[heldKey][key];
                if (action) {
                    executeChord(action);
                    e.preventDefault();
                    e.stopPropagation();
                }
            }
        }, true);

        window.addEventListener("keyup", (e) => {
            if (e.key.toLowerCase() === heldKey) heldKey = null;
        }, true);

        // Context menu
        const origGetMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const options = origGetMenuOptions.apply(this, arguments);
            const selected = getSelectedNodes(this);

            if (selected.length >= 2) {
                options.push(null);
                options.push({
                    content: "Align Nodes",
                    submenu: {
                        options: [
                            { content: "Align Left (Z+1)", callback: () => alignNodes(selected, "left") },
                            { content: "Align Right (Z+2)", callback: () => alignNodes(selected, "right") },
                            { content: "Align Top (Z+3)", callback: () => alignNodes(selected, "top") },
                            { content: "Align Bottom (Z+4)", callback: () => alignNodes(selected, "bottom") },
                            null,
                            { content: "Center Horizontal (X+1)", callback: () => alignNodes(selected, "centerH") },
                            { content: "Center Vertical (X+2)", callback: () => alignNodes(selected, "centerV") },
                            null,
                            { content: "Distribute Horizontal (C+1)", callback: () => distributeNodes(selected, "horizontal") },
                            { content: "Distribute Vertical (C+2)", callback: () => distributeNodes(selected, "vertical") },
                            null,
                            { content: "Match Width (V+1)", callback: () => matchSize(selected, "width") },
                            { content: "Match Height (V+2)", callback: () => matchSize(selected, "height") },
                            { content: "Match Size (V+3)", callback: () => matchSize(selected, "both") },
                        ]
                    }
                });
            }

            return options;
        };
    }
});
