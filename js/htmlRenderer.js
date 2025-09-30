import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.HTMLRenderer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "HTMLRenderer" && nodeData.category === "ETK/LLM/LiteLLM") {
            const updatePreviewHeight = function () {
                const display = this.htmlRendererDisplayEl;
                if (!display) {
                    return;
                }

                const heightWidget = this.widgets?.find?.((w) => w.name === "iframe_height");
                const rawHeight = typeof heightWidget?.value === "string" ? heightWidget.value.trim() : "";
                display.style.height = rawHeight || "auto";
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
                const node = this;
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

                updatePreviewHeight.call(this);

                const heightWidget = this.widgets?.find?.((w) => w.name === "iframe_height");
                if (heightWidget) {
                    const originalCallback = heightWidget.callback;
                    heightWidget.callback = function (...args) {
                        const callbackResult = originalCallback?.apply(this, args);
                        const nextValue = callbackResult !== undefined ? callbackResult : args[0];
                        updatePreviewHeight.call(node);
                        return nextValue;
                    };
                }

                return ret;
            };

            // Function to update display
            const updateDisplay = function (html_content) {
                if (html_content && html_content.length > 0) {
                    this.displayWidget.value = html_content;
                    app.graph.setDirtyCanvas(true);
                }

                updatePreviewHeight.call(this);
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

        if (nodeData.name === "HTMLServerLink" && nodeData.category === "ETK/LLM/LiteLLM") {
            const updateLinkDisplay = function (value) {
                const hasUrl = typeof value === "string" && value.startsWith("/lite-html/");
                const absoluteUrl = hasUrl ? new URL(value, window.location.origin).toString() : "";

                this.__liteHtmlUrl = hasUrl ? value : "";

                if (this.__liteHtmlButton) {
                    this.__liteHtmlButton.disabled = !hasUrl;
                }

                if (this.__liteHtmlLabel) {
                    if (hasUrl) {
                        this.__liteHtmlLabel.textContent = absoluteUrl;
                        this.__liteHtmlLabel.style.opacity = "1";
                    } else {
                        this.__liteHtmlLabel.textContent = value || "Run the node to generate a link.";
                        this.__liteHtmlLabel.style.opacity = value ? "1" : "0.7";
                    }
                }
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.gap = "6px";
                container.style.width = "100%";

                const button = document.createElement("button");
                button.type = "button";
                button.textContent = "Open HTML Page";
                button.disabled = true;
                button.addEventListener("click", () => {
                    if (!this.__liteHtmlUrl) {
                        return;
                    }
                    const absolute = new URL(this.__liteHtmlUrl, window.location.origin).toString();
                    window.open(absolute, "_blank", "noopener,noreferrer");
                });

                const label = document.createElement("div");
                label.style.wordBreak = "break-all";
                label.style.fontSize = "12px";
                label.style.opacity = "0.7";
                label.textContent = "Run the node to generate a link.";

                container.appendChild(button);
                container.appendChild(label);

                this.addDOMWidget("html_link", "display", container, {
                    serialize: false,
                    getValue: () => this.__liteHtmlUrl || "",
                    setValue: (value) => updateLinkDisplay.call(this, value),
                });

                this.__liteHtmlButton = button;
                this.__liteHtmlLabel = label;
                this.setSize([300, 120]);

                return ret;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const value = message?.string?.[0] ?? "";
                updateLinkDisplay.call(this, value);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (config) {
                onConfigure?.apply(this, arguments);
                const values = Array.isArray(config?.widgets_values) ? config.widgets_values : [];
                const stored = values.find((v) => typeof v === "string" && v.startsWith("/lite-html/")) || "";
                updateLinkDisplay.call(this, stored);
            };
        }
    },
});
