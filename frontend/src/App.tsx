import { useState, useRef, useCallback } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EMOTIONS = [
  { key: 'neutral', color: '#808080' },
  { key: 'happy', color: '#22c55e' },
  { key: 'sad', color: '#3b82f6' },
  { key: 'angry', color: '#ef4444' },
  { key: 'fearful', color: '#a855f7' },
  { key: 'calm', color: '#06b6d4' },
  { key: 'disgust', color: '#f59e0b' },
  { key: 'surprised', color: '#ec4899' },
]

const FEATURES = [
  {
    title: 'Real-time Detection',
    description: 'Stream audio via WebSocket and get emotion predictions every second. Sub-100ms latency on GPU.',
    icon: '⚡',
  },
  {
    title: 'Speech Transcription',
    description: 'Powered by OpenAI Whisper. Full transcript with timestamps alongside emotion analysis.',
    icon: '📝',
  },
  {
    title: 'Mamba Architecture',
    description: 'State-space model processes 480,000 tokens in O(n) time. No chunking hacks, no OOM errors.',
    icon: '🧠',
  },
  {
    title: '8 Emotion Classes',
    description: 'Trained on RAVDESS + CREMA-D datasets. Detects neutral, happy, sad, angry, fearful, calm, disgust, surprised.',
    icon: '🎭',
  },
  {
    title: 'REST + WebSocket API',
    description: 'Upload files via REST or stream live audio via WebSocket. Production-ready FastAPI backend.',
    icon: '🔌',
  },
  {
    title: 'Open Architecture',
    description: 'Swap models without breaking the pipeline. GRU fallback for CPU, Mamba SSM for GPU inference.',
    icon: '🔧',
  },
]

const INTEGRATIONS = [
  'Zoom', 'Google Meet', 'Microsoft Teams', 'Slack', 'Salesforce',
  'HubSpot', 'Twilio', 'Amazon Connect', 'Genesys', 'Five9',
  'Zendesk', 'Intercom', 'Freshdesk', 'RingCentral', 'Vonage',
]

/* ═══════════════════════════════════════════════════════════════════════════
   NAV
   ═══════════════════════════════════════════════════════════════════════════ */

function Nav() {
  return (
    <nav className="sticky top-0 z-50 bg-white/95 backdrop-blur-sm border-b border-border-light">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-6 lg:px-10 h-16">
        <a href="#" className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-ink flex items-center justify-center">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M2 12C2 6.5 6.5 2 12 2s10 4.5 10 10-4.5 10-10 10S2 17.5 2 12z" stroke="white" strokeWidth="2" />
              <path d="M8 12l3 3 5-6" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <span className="text-lg font-semibold tracking-tight text-ink">Mamba Audio</span>
        </a>

        <ul className="hidden md:flex items-center gap-8">
          {['Product', 'Features', 'Demo', 'Docs', 'Pricing'].map((item) => (
            <li key={item}>
              <a href={`#${item.toLowerCase()}`} className="text-sm font-medium text-text hover:text-ink transition-colors duration-200">
                {item}
              </a>
            </li>
          ))}
        </ul>

        <div className="flex items-center gap-3">
          <a href="#demo" className="hidden sm:inline-flex text-sm font-medium text-text-muted hover:text-ink transition-colors px-4 py-2">
            Try Demo
          </a>
          <a href="#demo" className="inline-flex items-center gap-2 text-sm font-medium text-white bg-ink hover:bg-black rounded-lg px-5 py-2.5 transition-colors duration-200">
            Get Started
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7" /></svg>
          </a>
        </div>
      </div>
    </nav>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   HERO
   ═══════════════════════════════════════════════════════════════════════════ */

function Hero() {
  return (
    <section className="relative overflow-hidden">
      <div className="absolute inset-0 opacity-[0.03]" style={{
        backgroundImage: 'linear-gradient(#333 1px, transparent 1px), linear-gradient(90deg, #333 1px, transparent 1px)',
        backgroundSize: '60px 60px',
      }} />

      <div className="relative max-w-7xl mx-auto px-6 lg:px-10 pt-28 pb-24 md:pt-40 md:pb-36">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="animate-fade-up inline-flex items-center gap-2 rounded-full border border-border bg-surface-light px-4 py-2 mb-10">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs font-medium text-text-muted tracking-wide uppercase">Now processing real-time audio</span>
          </div>

          {/* Headline */}
          <h1 className="animate-fade-up-1 text-5xl sm:text-6xl md:text-7xl font-bold text-ink leading-[1.1] tracking-tight">
            Understand every voice.
            <br />
            <span className="text-text-faint">Detect every emotion.</span>
          </h1>

          {/* Description */}
          <p className="animate-fade-up-2 mt-8 text-lg text-text-muted max-w-2xl mx-auto leading-relaxed">
            Real-time speech transcription and emotion detection powered by Mamba state-space models. Process hours of audio in seconds with O(n) efficiency.
          </p>

          {/* Dual CTAs */}
          <div className="animate-fade-up-3 mt-12 flex flex-col sm:flex-row items-center justify-center gap-4">
            <a href="#demo" className="inline-flex items-center gap-2 rounded-lg bg-ink text-white font-medium text-sm px-8 py-4 hover:bg-black transition-colors duration-200">
              Try for Free
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7" /></svg>
            </a>
            <a href="#features" className="inline-flex items-center gap-2 rounded-lg border border-border bg-white text-text font-medium text-sm px-8 py-4 hover:bg-surface hover:border-text-faint transition-all duration-200">
              See How It Works
            </a>
          </div>

          {/* Stats */}
          <div className="animate-fade-up-4 mt-20 flex items-center justify-center gap-10 sm:gap-16">
            {[
              { value: '<100ms', label: 'Latency' },
              { value: 'O(n)', label: 'Complexity' },
              { value: '8', label: 'Emotions' },
              { value: '50.8%', label: 'Val Accuracy' },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-xl sm:text-2xl font-bold text-ink" style={{ fontFamily: "'DM Mono', monospace" }}>
                  {stat.value}
                </div>
                <div className="text-xs text-text-muted mt-1.5 font-medium uppercase tracking-wider">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   SOCIAL PROOF MARQUEE
   ═══════════════════════════════════════════════════════════════════════════ */

function SocialProof() {
  return (
    <section className="border-y border-border-light bg-surface-light py-12 overflow-hidden">
      <p className="text-center text-xs font-medium text-text-faint uppercase tracking-widest mb-10">
        Integrates with the tools your team already uses
      </p>
      <div className="relative">
        <div className="flex animate-scroll-left">
          {[...INTEGRATIONS, ...INTEGRATIONS].map((name, i) => (
            <div key={`${name}-${i}`} className="flex-shrink-0 mx-8 flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-lg bg-surface flex items-center justify-center">
                <span className="text-xs font-bold text-text-muted">{name[0]}</span>
              </div>
              <span className="text-sm font-medium text-text-faint whitespace-nowrap">{name}</span>
            </div>
          ))}
        </div>
        <div className="absolute inset-y-0 left-0 w-24 bg-gradient-to-r from-surface-light to-transparent pointer-events-none" />
        <div className="absolute inset-y-0 right-0 w-24 bg-gradient-to-l from-surface-light to-transparent pointer-events-none" />
      </div>
    </section>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   FEATURES
   ═══════════════════════════════════════════════════════════════════════════ */

function Features() {
  return (
    <section id="features" className="py-28 md:py-40">
      <div className="max-w-7xl mx-auto px-6 lg:px-10">
        <div className="max-w-2xl mb-20">
          <p className="text-xs font-semibold text-brand uppercase tracking-widest mb-4">Platform</p>
          <h2 className="text-4xl sm:text-5xl font-bold text-ink tracking-tight leading-[1.1]">
            Built for real-time
            <br />
            <span className="text-text-faint">audio intelligence.</span>
          </h2>
          <p className="mt-6 text-base text-text-muted leading-relaxed max-w-xl">
            From raw waveform to emotion label in one forward pass. No chunking, no attention bottleneck, no memory limits.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map((feature) => (
            <div key={feature.title} className="group rounded-2xl border border-border-light bg-white p-8 hover:border-border hover:shadow-sm transition-all duration-300">
              <div className="w-12 h-12 rounded-xl bg-surface flex items-center justify-center text-xl mb-5 group-hover:bg-border-light transition-colors">
                {feature.icon}
              </div>
              <h3 className="text-base font-semibold text-ink mb-3">{feature.title}</h3>
              <p className="text-sm text-text-muted leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   ARCHITECTURE
   ═══════════════════════════════════════════════════════════════════════════ */

function Architecture() {
  const steps = [
    { label: 'Audio Input', detail: 'WAV / Stream' },
    { label: 'Patch Embed', detail: '10ms frames' },
    { label: 'Mamba SSM', detail: '6 layers', highlight: true },
    { label: 'Classifier', detail: '8 emotions', highlight: true },
    { label: 'Whisper', detail: 'Transcript' },
  ]

  return (
    <section className="py-28 md:py-40 bg-ink">
      <div className="max-w-7xl mx-auto px-6 lg:px-10">
        <div className="text-center mb-20">
          <p className="text-xs font-semibold text-brand-light uppercase tracking-widest mb-4">Architecture</p>
          <h2 className="text-4xl sm:text-5xl font-bold text-white tracking-tight leading-[1.1]">From waveform to insight</h2>
          <p className="mt-6 text-base text-text-faint max-w-xl mx-auto leading-relaxed">
            The Mamba state-space model processes entire audio sequences in linear time — no quadratic attention, no memory bottleneck.
          </p>
        </div>

        {/* Pipeline */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-0">
          {steps.map((step, i) => (
            <div key={step.label} className="flex items-center">
              <div className={`text-center px-6 py-5 rounded-2xl border min-w-[160px] ${step.highlight ? 'border-brand/40 bg-brand/10' : 'border-ink-muted/30 bg-ink-light'}`}>
                <div className="text-sm font-semibold text-white">{step.label}</div>
                <div className={`text-xs mt-1.5 font-medium ${step.highlight ? 'text-brand-light' : 'text-text-muted'}`} style={{ fontFamily: "'DM Mono', monospace" }}>
                  {step.detail}
                </div>
              </div>
              {i < steps.length - 1 && (
                <svg className="hidden sm:block w-10 h-4 text-text-muted mx-2 flex-shrink-0" viewBox="0 0 40 16" fill="none">
                  <path d="M0 8h34M28 2l6 6-6 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </div>
          ))}
        </div>

        {/* Benchmarks */}
        <div className="mt-24 grid grid-cols-1 sm:grid-cols-3 gap-5 max-w-3xl mx-auto">
          {[
            { model: 'Transformer', metric: 'O(n²)', memory: 'OOM at 4K tokens', highlight: false },
            { model: 'Mamba SSM', metric: 'O(n)', memory: '480K tokens, no OOM', highlight: true },
            { model: 'RNN/LSTM', metric: 'O(n)', memory: 'Gradient vanishing', highlight: false },
          ].map((item) => (
            <div key={item.model} className={`rounded-2xl p-7 text-center border ${item.highlight ? 'border-brand/40 bg-brand/10' : 'border-ink-muted/30 bg-ink-light'}`}>
              <div className={`text-sm font-semibold ${item.highlight ? 'text-brand-light' : 'text-text-faint'}`}>{item.model}</div>
              <div className={`text-3xl font-bold mt-3 ${item.highlight ? 'text-white' : 'text-text-faint'}`} style={{ fontFamily: "'DM Mono', monospace" }}>
                {item.metric}
              </div>
              <div className="text-xs text-text-muted mt-3">{item.memory}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   LIVE DEMO
   ═══════════════════════════════════════════════════════════════════════════ */

interface EmotionResult {
  dominant: string
  scores: Record<string, number>
}

interface TranscribeResult {
  session_id: string
  duration_sec: number
  transcript: string
  language: string
  emotions: EmotionResult
  processing_time_ms: number
}

function Demo() {
  const [result, setResult] = useState<TranscribeResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(async (file: File) => {
    setLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_URL}/api/transcribe`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const data: TranscribeResult = await res.json()
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to process audio')
    } finally {
      setLoading(false)
    }
  }, [])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }, [handleFile])

  const topEmotion = result ? EMOTIONS.find(e => e.key === result.emotions.dominant) : null

  return (
    <section id="demo" className="py-28 md:py-40 bg-surface-light">
      <div className="max-w-7xl mx-auto px-6 lg:px-10">
        <div className="text-center mb-16">
          <p className="text-xs font-semibold text-brand uppercase tracking-widest mb-4">Live Demo</p>
          <h2 className="text-4xl sm:text-5xl font-bold text-ink tracking-tight leading-[1.1]">Try it yourself</h2>
          <p className="mt-6 text-base text-text-muted max-w-xl mx-auto leading-relaxed">
            Upload a WAV file and watch the model detect emotions in real-time. The API is running on your local machine.
          </p>
        </div>

        <div className="max-w-3xl mx-auto">
          {/* Upload */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
            className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-16 text-center transition-all duration-200 ${dragOver ? 'border-brand bg-brand-bg' : 'border-border bg-white hover:border-text-faint hover:bg-surface'}`}
          >
            <input ref={fileRef} type="file" accept=".wav,.mp3,.flac,.ogg" onChange={onFileChange} className="hidden" />
            <div className="w-14 h-14 mx-auto rounded-xl bg-surface flex items-center justify-center mb-5">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#808080" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
              </svg>
            </div>
            <p className="text-sm font-medium text-text">
              Drop an audio file here, or <span className="text-brand">browse</span>
            </p>
            <p className="text-xs text-text-faint mt-2">WAV, MP3, FLAC, OGG supported</p>
          </div>

          {/* Loading */}
          {loading && (
            <div className="mt-10 text-center">
              <div className="inline-flex items-center gap-3">
                <div className="flex gap-1 items-end h-5">
                  {[0, 1, 2, 3, 4].map((i) => (
                    <div key={i} className="w-1 bg-brand rounded-full" style={{ animation: `pulse-bar 1s ease-in-out ${i * 0.15}s infinite`, height: '20px' }} />
                  ))}
                </div>
                <span className="text-sm font-medium text-text">Processing audio...</span>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mt-8 rounded-xl border border-red-200 bg-red-50 p-6">
              <p className="text-sm text-red-700"><span className="font-medium">Error:</span> {error}</p>
              <p className="text-xs text-red-500 mt-2">Make sure the backend is running on port 8000</p>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="mt-10 space-y-5">
              {/* Top emotion */}
              <div className="rounded-2xl border border-border-light bg-white p-8">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-sm font-semibold text-ink uppercase tracking-wider">Detected Emotion</h3>
                  <span className="text-xs text-text-faint" style={{ fontFamily: "'DM Mono', monospace" }}>{result.processing_time_ms.toFixed(0)}ms</span>
                </div>
                <div className="flex items-center gap-5">
                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center text-3xl" style={{ backgroundColor: `${topEmotion?.color}15` }}>
                    {result.emotions.dominant === 'happy' && '😊'}
                    {result.emotions.dominant === 'sad' && '😢'}
                    {result.emotions.dominant === 'angry' && '😠'}
                    {result.emotions.dominant === 'fearful' && '😨'}
                    {result.emotions.dominant === 'neutral' && '😐'}
                    {result.emotions.dominant === 'calm' && '😌'}
                    {result.emotions.dominant === 'disgust' && '🤢'}
                    {result.emotions.dominant === 'surprised' && '😮'}
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-ink capitalize">{result.emotions.dominant}</div>
                    <div className="text-sm text-text-faint mt-1">
                      {result.duration_sec}s audio &middot; {result.language !== 'unknown' ? result.language : 'auto-detect'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Emotion bars */}
              <div className="rounded-2xl border border-border-light bg-white p-8">
                <h3 className="text-sm font-semibold text-ink uppercase tracking-wider mb-6">Emotion Distribution</h3>
                <div className="space-y-4">
                  {Object.entries(result.emotions.scores)
                    .sort(([, a], [, b]) => b - a)
                    .map(([emotion, score]) => {
                      const emotionData = EMOTIONS.find(e => e.key === emotion)
                      return (
                        <div key={emotion} className="flex items-center gap-4">
                          <span className="text-sm font-medium text-text w-24 capitalize">{emotion}</span>
                          <div className="flex-1 h-2.5 bg-surface rounded-full overflow-hidden">
                            <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.max(score * 100, 1)}%`, backgroundColor: emotionData?.color || '#808080' }} />
                          </div>
                          <span className="text-sm font-medium text-text-muted w-14 text-right" style={{ fontFamily: "'DM Mono', monospace" }}>
                            {(score * 100).toFixed(1)}%
                          </span>
                        </div>
                      )
                    })}
                </div>
              </div>

              {/* Transcript */}
              {result.transcript && (
                <div className="rounded-2xl border border-border-light bg-white p-8">
                  <h3 className="text-sm font-semibold text-ink uppercase tracking-wider mb-4">Transcript</h3>
                  <p className="text-sm text-text leading-relaxed">{result.transcript}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   USE CASES
   ═══════════════════════════════════════════════════════════════════════════ */

function UseCases() {
  const cases = [
    {
      department: 'Sales',
      title: 'Call sentiment analysis',
      description: 'Detect customer frustration in real-time. Alert managers when calls escalate. Improve close rates with emotional intelligence.',
      metric: '23%',
      metricLabel: 'Better close rates',
    },
    {
      department: 'Support',
      title: 'Customer emotion tracking',
      description: 'Route angry customers to senior agents. Track satisfaction trends. Reduce churn with proactive intervention.',
      metric: '40%',
      metricLabel: 'Faster resolution',
    },
    {
      department: 'HR',
      title: 'Interview analysis',
      description: 'Assess candidate engagement and stress levels. Standardize interview evaluation. Remove unconscious bias.',
      metric: '3x',
      metricLabel: 'More consistent',
    },
  ]

  return (
    <section className="py-28 md:py-40">
      <div className="max-w-7xl mx-auto px-6 lg:px-10">
        <div className="text-center mb-20">
          <p className="text-xs font-semibold text-brand uppercase tracking-widest mb-4">Use Cases</p>
          <h2 className="text-4xl sm:text-5xl font-bold text-ink tracking-tight leading-[1.1]">
            Every team benefits from
            <br />
            <span className="text-text-faint">understanding emotion.</span>
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {cases.map((item) => (
            <div key={item.department} className="rounded-2xl border border-border-light bg-white p-10 hover:border-border hover:shadow-sm transition-all duration-300">
              <span className="inline-block text-xs font-semibold text-brand uppercase tracking-widest mb-5">{item.department}</span>
              <h3 className="text-xl font-semibold text-ink mb-3">{item.title}</h3>
              <p className="text-sm text-text-muted leading-relaxed mb-8">{item.description}</p>
              <div className="pt-6 border-t border-border-light">
                <div className="text-4xl font-bold text-ink" style={{ fontFamily: "'DM Mono', monospace" }}>{item.metric}</div>
                <div className="text-xs text-text-faint mt-1 font-medium uppercase tracking-wider">{item.metricLabel}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   CTA
   ═══════════════════════════════════════════════════════════════════════════ */

function CTA() {
  return (
    <section className="py-28 md:py-40 bg-ink">
      <div className="max-w-3xl mx-auto text-center px-6">
        <h2 className="text-4xl sm:text-5xl font-bold text-white tracking-tight leading-[1.1]">
          Ready to hear what your
          <br />
          <span className="text-text-muted">audio is really saying?</span>
        </h2>
        <p className="mt-6 text-base text-text-faint max-w-xl mx-auto leading-relaxed">
          Start detecting emotions in your audio today. Self-hosted, private, and fully under your control.
        </p>
        <div className="mt-12 flex flex-col sm:flex-row items-center justify-center gap-4">
          <a href="#demo" className="inline-flex items-center gap-2 rounded-lg bg-white text-ink font-medium text-sm px-8 py-4 hover:bg-surface transition-colors duration-200">
            Try the Demo
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7" /></svg>
          </a>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 rounded-lg border border-text-muted text-text-faint font-medium text-sm px-8 py-4 hover:bg-ink-light transition-all duration-200">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" /></svg>
            View on GitHub
          </a>
        </div>
      </div>
    </section>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   FOOTER
   ═══════════════════════════════════════════════════════════════════════════ */

function Footer() {
  return (
    <footer className="border-t border-border-light bg-white py-14">
      <div className="max-w-7xl mx-auto px-6 lg:px-10">
        <div className="flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-md bg-ink flex items-center justify-center">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                <path d="M2 12C2 6.5 6.5 2 12 2s10 4.5 10 10-4.5 10-10 10S2 17.5 2 12z" stroke="white" strokeWidth="2" />
                <path d="M8 12l3 3 5-6" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <span className="text-sm font-semibold text-ink">Mamba Audio AI</span>
          </div>

          <div className="flex items-center gap-8">
            {['Product', 'Features', 'Demo', 'Docs', 'GitHub'].map((item) => (
              <a key={item} href="#" className="text-xs font-medium text-text-faint hover:text-text transition-colors">{item}</a>
            ))}
          </div>

          <p className="text-xs text-text-faint">Built with Mamba SSM + Whisper + FastAPI</p>
        </div>
      </div>
    </footer>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   APP
   ═══════════════════════════════════════════════════════════════════════════ */

function App() {
  return (
    <div className="min-h-screen bg-white">
      <Nav />
      <Hero />
      <SocialProof />
      <Features />
      <Architecture />
      <Demo />
      <UseCases />
      <CTA />
      <Footer />
    </div>
  )
}

export default App
