import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "IXIWORKS.LoadImageList",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "UtilLoadImageList") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);

            const self = this;
            const countWidget = this.widgets.find((w) => w.name === "count");
            if (!countWidget) return;

            const origCallback = countWidget.callback;
            countWidget.callback = function (value) {
                if (origCallback) origCallback.call(this, value);
                self._updateImageDropdowns(value);
            };

            this._updateImageDropdowns(countWidget.value);
        };

        nodeType.prototype._updateImageDropdowns = function (count) {
            for (const w of this.widgets) {
                if (!w.name.startsWith("image_")) continue;
                const idx = parseInt(w.name.split("_")[1]);
                if (idx <= count) {
                    w.hidden = false;
                    w.computeSize = null;
                } else {
                    w.hidden = true;
                    w.computeSize = () => [0, -4];
                }
            }
            this.setSize(this.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const countWidget = this.widgets && this.widgets.find((w) => w.name === "count");
            if (countWidget) {
                this._updateImageDropdowns(countWidget.value);
            }
        };
    }
});
