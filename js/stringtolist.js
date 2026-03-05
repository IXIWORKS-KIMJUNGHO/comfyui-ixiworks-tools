import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "IXIWORKS.StringToList",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "UtilStringToList") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);

            const self = this;
            const countWidget = this.widgets.find((w) => w.name === "count");
            if (!countWidget) return;

            const origCallback = countWidget.callback;
            countWidget.callback = function (value) {
                if (origCallback) origCallback.call(this, value);
                self._updateVisiblePrompts(value);
            };

            // Initial sync
            this._updateVisiblePrompts(countWidget.value);
        };

        nodeType.prototype._updateVisiblePrompts = function (count) {
            for (const w of this.widgets) {
                if (!w.name.startsWith("prompt_")) continue;
                const idx = parseInt(w.name.split("_")[1]);
                if (idx <= count) {
                    w.type = "customtext";
                    w.computeSize = undefined;
                } else {
                    w.type = "hidden";
                    w.computeSize = () => [0, -4];
                }
            }
            this.setSize(this.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };

        // Restore after graph load
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const countWidget = this.widgets && this.widgets.find((w) => w.name === "count");
            if (countWidget) {
                this._updateVisiblePrompts(countWidget.value);
            }
        };
    }
});
