import { useEffect, useState } from "react";
import CommandCenter from "./pages/CommandCenter";
import { animate, stagger } from "animejs";
import "./App.css";

function App() {
  const [systemEntered, setSystemEntered] = useState(false);
  useEffect(() => {
    // Main entrance animation
    animate(".hero-content", {
      opacity: [0, 1],
      translateY: [40, 0],
      duration: 1200,
      ease: "outExpo",
    });

    // Stagger the system capabilities
    animate(".capability", {
      opacity: [0, 1],
      translateY: [20, 0],
      delay: stagger(120, { start: 700 }),
      duration: 700,
      ease: "outQuad",
    });

    // Animate the small status indicator
    animate(".status-dot", {
      scale: [1, 1.5, 1],
      opacity: [1, 0.5, 1],
      duration: 1600,
      loop: true,
      ease: "inOutSine",
    });
  }, []);

if (systemEntered) {
  return (
    <CommandCenter
      onBack={() => setSystemEntered(false)}
    />
  );
}

  return (
    <main className="nirmaan-page">
      <div className="grid-background"></div>

      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"></span>
          NIRMAAN<span className="brand-ai"> AI</span>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          SYSTEM ONLINE
        </div>
      </header>

      <section className="hero">
        <div className="hero-content">
          <p className="eyebrow">URBAN WASTE INTELLIGENCE</p>

          <h1>
            WASTE
            <br />
            <span>INTELLIGENCE.</span>
          </h1>

          <p className="hero-description">
            Detect waste. Understand its impact. Prioritize action.
            <br />
            Nirmaan turns waste data into actionable intelligence.
          </p>

          <button
            className="enter-button"
            onClick={() => setSystemEntered(true)}
          >
            <span>ENTER SYSTEM</span>
            <span className="arrow">→</span>
          </button>
        </div>

        <div className="hero-visual">
          <div className="orbital orbital-one"></div>
          <div className="orbital orbital-two"></div>

          <div className="core">
            <span>N</span>
          </div>

          <div className="data-point point-one"></div>
          <div className="data-point point-two"></div>
          <div className="data-point point-three"></div>
        </div>
      </section>

      <section className="capabilities">
        <div className="capability">
          <span className="capability-number">01</span>
          <span>DETECT</span>
        </div>

        <div className="capability">
          <span className="capability-number">02</span>
          <span>ANALYZE</span>
        </div>

        <div className="capability">
          <span className="capability-number">03</span>
          <span>PRIORITIZE</span>
        </div>

        <div className="capability">
          <span className="capability-number">04</span>
          <span>ACT</span>
        </div>
      </section>

      <footer className="footer">
        <span>NIRMAAN AI / 2026</span>
        <span>AI-POWERED WASTE INTELLIGENCE PLATFORM</span>
      </footer>
    </main>
  );
}


export default App;