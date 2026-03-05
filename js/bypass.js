import { app } from "../../scripts/app.js";

// ComfyUI node modes
const MODE_ALWAYS = 0;
const MODE_MUTE = 2;
const MODE_BYPASS = 4;

function setUpstreamMode(node, bypass) {
    if (!node.graph) return;
    if (!node.inputs || node.inputs.length === 0) return;

    const link = node.inputs[0].link;
    if (link == null) return;
    const linkInfo = node.graph.links[link];
    if (!linkInfo) return;
    const sourceNode = node.graph.getNodeById(linkInfo.origin_id);
    if (!sourceNode) return;

    if (bypass) {
        // Only bypass processing nodes (nodes that have inputs).
        // Source nodes (Load Image etc.) have no inputs so bypass
        // would produce no output → error. Leave them running.
        if (sourceNode.inputs && sourceNode.inputs.length > 0) {
            sourceNode.mode = MODE_BYPASS;
        }
    } else {
        sourceNode.mode = MODE_ALWAYS;
    }

    node.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "IXIWORKS.Bypass",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "UtilBypass") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);

            const bypassWidget = this.widgets.find((w) => w.name === "bypass");
            if (!bypassWidget) return;

            const self = this;
            const origCallback = bypassWidget.callback;
            bypassWidget.callback = function (value) {
                setUpstreamMode(self, value);
                if (origCallback) origCallback.call(this, value);
            };
        };

        // Restore upstream node mode after graph load
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);

            // Defer to next frame so all nodes are fully loaded
            const self = this;
            requestAnimationFrame(() => {
                const bypassWidget = self.widgets && self.widgets.find((w) => w.name === "bypass");
                if (bypassWidget && bypassWidget.value) {
                    setUpstreamMode(self, bypassWidget.value);
                }
            });
        };

        // Sync mode when connection changes
        const origConnChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot, isConnect, linkInfo) {
            if (origConnChange) origConnChange.apply(this, arguments);

            const bypassWidget = this.widgets && this.widgets.find((w) => w.name === "bypass");
            if (!bypassWidget) return;

            if (slotType === 1 && slot === 0) {
                if (isConnect && bypassWidget.value) {
                    // New connection + bypass is on → set source to bypass
                    setUpstreamMode(this, true);
                }
                if (!isConnect && linkInfo) {
                    // Disconnected → restore source node to normal
                    const sourceNode = this.graph.getNodeById(linkInfo.origin_id);
                    if (sourceNode) {
                        sourceNode.mode = MODE_ALWAYS;  // restore to normal
                        this.graph.setDirtyCanvas(true, true);
                    }
                }
            }
        };
    }
});
