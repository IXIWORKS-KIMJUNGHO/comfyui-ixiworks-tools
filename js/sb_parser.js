import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "IXIWORKS.SBJsonParser",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SBJsonParser") return;

        const PREVIEW_LINE_HEIGHT = 16;
        const PREVIEW_PADDING = 10;
        const PREVIEW_FONT = "12px monospace";
        const MIN_NODE_WIDTH = 280;

        // Capture execution result for preview
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (data) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);
            if (data?.preview) {
                this._previewLines = (data.preview[0] || "").split("\n");
                // Expand node to fit preview
                const previewH =
                    this._previewLines.length * PREVIEW_LINE_HEIGHT +
                    PREVIEW_PADDING * 2 +
                    8;
                const baseH = this._baseHeight || this.size[1];
                if (!this._baseHeight) this._baseHeight = this.size[1];
                this.size[0] = Math.max(this.size[0], MIN_NODE_WIDTH);
                this.size[1] = baseH + previewH;
                app.graph.setDirtyCanvas(true, true);
            }
        };

        // Draw preview at bottom of node
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origOnDrawForeground)
                origOnDrawForeground.apply(this, arguments);
            if (!this._previewLines) return;

            const lines = this._previewLines;
            const previewH =
                lines.length * PREVIEW_LINE_HEIGHT + PREVIEW_PADDING * 2;
            const startY = this.size[1] - previewH - 4;
            const boxX = 6;
            const boxW = this.size[0] - 12;

            // Background
            ctx.fillStyle = "rgba(20, 20, 20, 0.85)";
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(boxX, startY, boxW, previewH, 6);
            } else {
                ctx.rect(boxX, startY, boxW, previewH);
            }
            ctx.fill();

            // Border
            ctx.strokeStyle = "rgba(80, 80, 80, 0.6)";
            ctx.lineWidth = 1;
            ctx.stroke();

            // Text
            ctx.font = PREVIEW_FONT;
            ctx.textBaseline = "top";

            lines.forEach((line, i) => {
                const y =
                    startY + PREVIEW_PADDING + i * PREVIEW_LINE_HEIGHT;

                if (line.startsWith("Scene")) {
                    ctx.fillStyle = "#e0e0e0";
                } else if (line.startsWith("  Cut")) {
                    ctx.fillStyle = "#999";
                } else if (line.startsWith("Characters:")) {
                    ctx.fillStyle = "#8bc78b";
                } else if (line.startsWith("Error")) {
                    ctx.fillStyle = "#e07070";
                } else {
                    ctx.fillStyle = "#bbb";
                }

                // Clip text to node width
                const maxW = boxW - PREVIEW_PADDING * 2;
                ctx.fillText(line, boxX + PREVIEW_PADDING, y, maxW);
            });
        };
    },
});
