import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.HTMLRenderer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "HTMLRenderer" && nodeData.category === "ETK/LLM/LiteLLM") {
            // Node Created
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = onNodeCreated
                    ? onNodeCreated.apply(this, arguments)
                    : undefined;

                // Only create the html_content widget if it doesn't exist
                if (!this.widgets.find(w => w.name === "html_content")) {
                    ComfyWidgets.STRING(
                        this,
                        "html_content",
                        ["STRING", {
                            default: "",
                            placeholder: "HTML content..."
                        }],
                        app
                    );
                }

                // Create HTML display widget
                const displayEl = document.createElement("div");
                displayEl.style.width = "100%";
                displayEl.style.height = "auto";
                displayEl.style.overflow = "auto";
                displayEl.style.border = "1px solid #ddd";
                displayEl.style.borderRadius = "5px";
                displayEl.style.padding = "0px";
                displayEl.style.backgroundColor = "#FFFFFF";

                this.displayWidget = this.addDOMWidget("html_display", "display", displayEl, {
                    serialize: false,
                    getValue() {
                        return displayEl.innerHTML;
                    },
                    setValue(v) {
                        displayEl.innerHTML = v;
                    },
                });

                this.setSize([350, 400]);

                return ret;
            };

            // Function to update display
            const updateDisplay = function (html_content) {
                if (html_content && html_content.length > 0) {
                    this.displayWidget.value = html_content;
                    app.graph.setDirtyCanvas(true);
                }
            };

            // onExecuted
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                updateDisplay.call(this, message.string[0]);
            };

            // onConfigure
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (config) {
                onConfigure?.apply(this, arguments);
                if (config?.widgets_values?.length) {
                    updateDisplay.call(this, config.widgets_values[0]);
                }
            };
        }
    },
});
