import { useEffect, useRef, useState } from "react";
import { animate, stagger } from "animejs";
import "./CommandCenter.css";

function CommandCenter({ onBack }) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const uploadZoneRef = useRef(null);

  // Release image preview URLs when the image changes or the page unmounts.
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // Validate the image and prepare it for analysis.
  const processImage = (file) => {
    if (!file) return;

    const allowedTypes = ["image/jpeg", "image/png"];
    const allowedExtensions = /\.(jpe?g|png)$/i;

    if (
      !allowedTypes.includes(file.type) &&
      !allowedExtensions.test(file.name)
    ) {
      setAnalysisError(
        "Unsupported file. Please select a JPG or PNG image."
      );
      setIsDragging(false);
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setAnalysisError(
        "Image exceeds 10 MB. Please choose a smaller file."
      );
      setIsDragging(false);
      return;
    }

    if (file.size === 0) {
      setAnalysisError("This file is empty. Please choose another image.");
      setIsDragging(false);
      return;
    }

    setAnalysisError(null);
    setAnalysisResult(null);
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setIsDragging(false);

    // Anime.js feedback when an image is selected.
    if (uploadZoneRef.current) {
      animate(uploadZoneRef.current, {
        scale: [1, 1.015, 1],
        duration: 350,
        ease: "outQuad",
      });
    }
  };

  const handleImageSelect = (event) => {
    const file = event.target.files?.[0];

    // Let the user select the same file again if needed.
    event.target.value = "";

    processImage(file);
  };

  // Drag-and-drop handlers.
  const handleDragOver = (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";

    if (!isDragging) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (event) => {
    event.preventDefault();

    // Ignore movement between children inside the upload zone.
    if (
      event.relatedTarget &&
      event.currentTarget.contains(event.relatedTarget)
    ) {
      return;
    }

    setIsDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);

    const file = event.dataTransfer.files?.[0];

    if (!file) {
      setAnalysisError("No file detected. Please drop an image.");
      return;
    }

    processImage(file);
  };

  const handleAnalyze = async () => {
    if (!selectedImage || isAnalyzing) return;

    setIsAnalyzing(true);
    setAnalysisResult(null);
    setAnalysisError(null);

    // Animate the beginning of the scan.
    if (uploadZoneRef.current) {
      animate(uploadZoneRef.current, {
        opacity: [0.75, 1],
        duration: 450,
        ease: "outQuad",
      });
    }

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);

      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/api/v1/waste/analyze-image`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        let message = `Analysis failed (${response.status}).`;

        try {
          const errorData = await response.json();
          message =
            errorData.detail || errorData.message || message;
        } catch {
          // Keep the fallback error if the response isn't JSON.
        }

        throw new Error(message);
      }

      const result = await response.json();

      // Verify the response has the fields needed by the UI.
      if (
        !result ||
        typeof result.predicted_class !== "string" ||
        typeof result.confidence !== "number"
      ) {
        throw new Error("The AI backend returned an unexpected response.");
      }

      console.log("NIRMAAN AI RESULT:", result);
      setAnalysisResult(result);

      // Animate the result when it appears.
      requestAnimationFrame(() => {
        const resultElement = document.querySelector(".analysis-result");

        if (resultElement) {
          animate(resultElement, {
            opacity: [0, 1],
            translateY: [12, 0],
            duration: 450,
            ease: "outExpo",
          });
        }
      });
    } catch (error) {
      console.error("Waste analysis failed:", error);

      setAnalysisError(
        error.message ||
          "Unable to analyze the image. Please check that the AI backend is running."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleChangeImage = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setAnalysisResult(null);
    setAnalysisError(null);
    setIsDragging(false);
  };

  // Entrance animations and live indicator.
  useEffect(() => {
    animate(
      ".command-header, .metric-card, .intel-panel, .pipeline-item",
      {
        opacity: [0, 1],
        translateY: [20, 0],
        delay: stagger(80),
        duration: 700,
        ease: "outExpo",
      }
    );

    animate(".map-point", {
      scale: [0, 1],
      opacity: [0, 1],
      delay: stagger(120),
      duration: 500,
      ease: "outBack",
    });

    const liveAnimation = animate(".live-dot", {
      scale: [1, 1.5, 1],
      opacity: [1, 0.5, 1],
      duration: 1500,
      loop: true,
      ease: "inOutSine",
    });

    return () => {
      liveAnimation.pause();
    };
  }, []);

  return (
    <main className="command-page">
      <header className="command-header">
        <button className="command-brand" onClick={onBack}>
          <span className="brand-mark"></span>
          NIRMAAN<span> AI</span>
        </button>

        <div className="command-status">
          <span className="live-dot"></span>
          SYSTEM ONLINE
        </div>
      </header>

      <section className="command-title">
        <div>
          <p className="section-label">NIRMAAN / COMMAND CENTER</p>
          <h1>CITY INTELLIGENCE</h1>
        </div>

        <div className="command-date">
          <span>LIVE MONITORING</span>
          <strong>2026</strong>
        </div>
      </section>

      <section className="metrics">
        <div className="metric-card">
          <span className="metric-label">WASTE DETECTED</span>
          <strong>1,284</strong>
          <small>+12.4% THIS WEEK</small>
        </div>

        <div className="metric-card">
          <span className="metric-label">PRIORITY ZONES</span>
          <strong>17</strong>
          <small>5 REQUIRE ACTION</small>
        </div>

        <div className="metric-card">
          <span className="metric-label">CLEANUP STATUS</span>
          <strong>82%</strong>
          <small>OPERATIONS ACTIVE</small>
        </div>

        <div className="metric-card">
          <span className="metric-label">AI CONFIDENCE</span>
          <strong>91.7%</strong>
          <small>MODEL STATUS: STABLE</small>
        </div>
      </section>

      <section className="intelligence-grid">
        <div className="intel-panel activity-map">
          <div className="panel-heading">
            <div>
              <span className="section-label">01 / DETECTION</span>
              <h2>WASTE ACTIVITY</h2>
            </div>

            <span className="panel-status">LIVE</span>
          </div>

          <div className="map-area">
            <div className="map-grid"></div>

            <div className="map-line line-a"></div>
            <div className="map-line line-b"></div>
            <div className="map-line line-c"></div>

            <span className="map-point point-a"></span>
            <span className="map-point point-b"></span>
            <span className="map-point point-c"></span>
            <span className="map-point point-d"></span>
            <span className="map-point point-e"></span>
            <span className="map-point point-f"></span>

            <div className="map-center">
              <span>LIVE</span>
            </div>
          </div>

          <div className="map-footer">
            <span>DETECTION NETWORK</span>
            <span>6 ACTIVE NODES</span>
          </div>
        </div>

        <div className="intel-panel analyze-panel">
          <div className="panel-heading">
            <div>
              <span className="section-label">02 / ANALYSIS</span>
              <h2>ANALYZE WASTE</h2>
            </div>
          </div>

          <div
            ref={uploadZoneRef}
            className={`upload-zone${isDragging ? " is-dragging" : ""}`}
            onDragEnter={handleDragOver}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {!selectedImage ? (
              <>
                <div className="upload-icon">+</div>

                <h3>
                  {isDragging ? "DROP IMAGE HERE" : "DROP WASTE IMAGE"}
                </h3>

                <p>
                  Drag and drop an image here, or select one
                  <br />
                  to classify waste and generate intelligence.
                </p>

                <label className="upload-button">
                  SELECT IMAGE
                  <span>→</span>
                  <input
                    type="file"
                    accept="image/png,image/jpeg,.jpg,.jpeg,.png"
                    onChange={handleImageSelect}
                    hidden
                  />
                </label>

                <small className="upload-format">
                  JPG / PNG • MAX 10 MB
                </small>
              </>
            ) : isAnalyzing ? (
              <>
                <div className="scan-preview-wrapper">
                  <img
                    src={previewUrl}
                    alt="Scanning waste"
                    className="scan-preview"
                  />

                  <div className="scan-line"></div>

                  <div className="scan-corner top-left"></div>
                  <div className="scan-corner top-right"></div>
                  <div className="scan-corner bottom-left"></div>
                  <div className="scan-corner bottom-right"></div>
                </div>

                <div className="scan-status">
                  <span className="ready-dot"></span>
                  AI SCANNING
                </div>

                <div className="scan-progress">
                  <div className="scan-progress-bar"></div>
                </div>

                <div className="scan-info">
                  <span>ANALYZING VISUAL FEATURES</span>
                  <span>CLASSIFYING WASTE</span>
                </div>
              </>
            ) : analysisResult ? (
              <div className="analysis-result">
                <div className="result-status">
                  <span className="ready-dot"></span>
                  AI ANALYSIS COMPLETE
                </div>

                <div className="result-class">
                  <span className="result-label">PREDICTED WASTE</span>
                  <h3>{analysisResult.predicted_class.toUpperCase()}</h3>
                </div>

                <div className="result-confidence">
                  <span>CONFIDENCE</span>
                  <strong>
                    {(analysisResult.confidence * 100).toFixed(2)}%
                  </strong>
                </div>

                <div className="result-details">
                  <div>
                    <span>CATEGORY</span>
                    <strong>{analysisResult.category ?? "Unknown"}</strong>
                  </div>

                  <div>
                    <span>RECYCLABLE</span>
                    <strong>
                      {analysisResult.recyclable ? "YES" : "NO"}
                    </strong>
                  </div>
                </div>

                <div className="result-info">
                  <span>DISPOSAL METHOD</span>
                  <p>
                    {analysisResult.disposal_method ?? "Not provided"}
                  </p>
                </div>

                <div className="result-info">
                  <span>ENVIRONMENTAL IMPACT</span>
                  <p>
                    {analysisResult.environmental_impact ?? "Not provided"}
                  </p>
                </div>

                <button
                  className="upload-button analyze-button"
                  onClick={handleChangeImage}
                >
                  ANALYZE ANOTHER
                  <span>→</span>
                </button>
              </div>
            ) : (
              <>
                <div className="image-preview-wrapper">
                  <img
                    src={previewUrl}
                    alt="Selected waste"
                    className="image-preview"
                  />
                </div>

                <div className="image-ready">
                  <span className="ready-dot"></span>
                  IMAGE READY
                </div>

                <p className="selected-file">{selectedImage.name}</p>

                <button
                  className="upload-button analyze-button"
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                >
                  ANALYZE WITH AI
                  <span>→</span>
                </button>

                <button
                  className="change-image"
                  onClick={handleChangeImage}
                  disabled={isAnalyzing}
                >
                  CHANGE IMAGE
                </button>
              </>
            )}

            {analysisError && (
              <p className="analysis-error" role="alert">
                {analysisError}
              </p>
            )}
          </div>

          <div className="analysis-note">
            <span>AI ENGINE</span>
            <strong>EFFNET-B0 / CNN</strong>
          </div>
        </div>
      </section>

      <section className="pipeline">
        <div className="pipeline-heading">
          <span className="section-label">SYSTEM PIPELINE</span>
          <span>REAL-TIME WASTE OPERATIONS</span>
        </div>

        <div className="pipeline-items">
          <div className="pipeline-item active">
            <span>01</span>
            DETECT
          </div>

          <div className="pipeline-arrow">→</div>

          <div className="pipeline-item active">
            <span>02</span>
            ANALYZE
          </div>

          <div className="pipeline-arrow">→</div>

          <div className="pipeline-item">
            <span>03</span>
            PRIORITIZE
          </div>

          <div className="pipeline-arrow">→</div>

          <div className="pipeline-item">
            <span>04</span>
            ASSIGN
          </div>

          <div className="pipeline-arrow">→</div>

          <div className="pipeline-item">
            <span>05</span>
            VERIFY
          </div>
        </div>
      </section>

      <footer className="command-footer">
        <span>NIRMAAN AI / URBAN WASTE INTELLIGENCE</span>
        <span>BUILDING CLEANER CITIES THROUGH DATA</span>
      </footer>
    </main>
  );
}

export default CommandCenter;
