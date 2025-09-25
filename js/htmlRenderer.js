import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.HTMLRenderer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "HTMLRenderer" && nodeData.category === "ETK/LLM/LiteLLM") {
            const applyIframeHeight = function () {
                if (!this.htmlRendererDisplayEl) {
                    return;
                }

                const heightWidget = this.widgets?.find?.((w) => w.name === "iframe_height");
                const rawValue = typeof heightWidget?.value === "string" ? heightWidget.value.trim() : "";
                this.htmlRendererDisplayEl.style.height = rawValue || "auto";
                app.graph.setDirtyCanvas(true);
            };

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
                displayEl.style.minHeight = "200px";
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

                this.htmlRendererDisplayEl = displayEl;

                this.setSize([350, 400]);

                applyIframeHeight.call(this);

                return ret;
            };

            // Function to update display
            const updateDisplay = function (html_content) {
                if (typeof html_content === "string") {
                    this.displayWidget.value = html_content;
                }

                applyIframeHeight.call(this);
            };

            // onExecuted
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const html = message?.string?.[0];
                updateDisplay.call(this, html);
            };

            // onConfigure
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (config) {
                onConfigure?.apply(this, arguments);
                const configuredHtml = config?.widgets_values?.[0];
                if (typeof configuredHtml === "string") {
                    updateDisplay.call(this, configuredHtml);
                } else {
                    applyIframeHeight.call(this);
                }
            };

            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function (widget, value, ...args) {
                const ret = onWidgetChanged?.apply(this, [widget, value, ...args]);
                if (widget?.name === "iframe_height") {
                    applyIframeHeight.call(this);
                }
                return ret;
            };
        }
    },
});
