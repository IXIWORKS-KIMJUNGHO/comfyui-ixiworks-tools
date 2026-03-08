import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "IXIWORKS.SBConcatStrings",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SBConcatStrings") return;

        // Remove optional string inputs from frontend definition
        // to prevent auto-creation of 20 slots (Python still declares them for API)
        if (nodeData.input?.optional) {
            for (let i = 1; i <= 20; i++) {
                delete nodeData.input.optional[`string_${i}`];
            }
        }

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);

            const self = this;
            const numWidget = this.widgets.find((w) => w.name === "num_inputs");
            if (!numWidget) return;

            numWidget.callback = function (value) {
                // Find highest connected string input
                let maxConnected = 0;
                for (const inp of self.inputs || []) {
                    if (inp.name.startsWith("string_") && inp.link != null) {
                        const idx = parseInt(inp.name.split("_")[1]);
                        maxConnected = Math.max(maxConnected, idx);
                    }
                }

                // Clamp: prevent removing connected inputs
                if (value < maxConnected) {
                    numWidget.value = maxConnected;
                    return;
                }

                self._syncStringInputs(value);
            };

            // Initial sync
            this._syncStringInputs(numWidget.value);
        };

        nodeType.prototype._syncStringInputs = function (target) {
            const existing = (this.inputs || []).filter((i) =>
                i.name.startsWith("string_")
            );
            let count = existing.length;

            // Remove excess from the end
            while (count > target) {
                const idx = this.inputs.findIndex(
                    (inp) => inp.name === `string_${count}`
                );
                if (idx !== -1) this.removeInput(idx);
                count--;
            }

            // Add missing
            while (count < target) {
                count++;
                this.addInput(`string_${count}`, "STRING");
            }

            this.setSize(this.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };

        // Restore after graph/workflow load
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const numWidget =
                this.widgets && this.widgets.find((w) => w.name === "num_inputs");
            if (numWidget) {
                this._syncStringInputs(numWidget.value);
            }
        };
    },
});
