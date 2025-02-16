"use client"

import type React from "react"
import { useEffect, useRef } from "react"

interface YggdrasilProps {
  className?: string
}

const Yggdrasil: React.FC<YggdrasilProps> = ({ className = "" }) => {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // Create floating wisps
    const createWisps = () => {
      const wispsContainer = document.createElement("div")
      wispsContainer.className = "yggdrasil-wisps"
      for (let i = 0; i < 30; i++) {
        const wisp = document.createElement("div")
        wisp.className = "wisp"
        wisp.style.setProperty("--delay", `${Math.random() * 10}s`)
        wisp.style.setProperty("--duration", `${Math.random() * 5 + 5}s`)
        wisp.style.setProperty("--scale", `${Math.random() * 0.5 + 0.5}`)
        wispsContainer.appendChild(wisp)
      }
      container.appendChild(wispsContainer)
    }

    createWisps()

    const handleMouseMove = (e: MouseEvent) => {
      const { clientX, clientY } = e
      const { width, height } = container.getBoundingClientRect()

      const x = (clientX / width - 0.5) * 40
      const y = (clientY / height - 0.5) * 40

      container.style.setProperty("--mouse-x", `${x}px`)
      container.style.setProperty("--mouse-y", `${y}px`)
    }

    window.addEventListener("mousemove", handleMouseMove)
    return () => window.removeEventListener("mousemove", handleMouseMove)
  }, [])

  return (
    <div ref={containerRef} className={`yggdrasil-background ${className}`.trim()}>
      <div className="yggdrasil-base" />
      <div className="yggdrasil-glow-overlay" />
      <div className="yggdrasil-mist-layer" />
      <div className="yggdrasil-cosmic-overlay" />
    </div>
  )
}

export default Yggdrasil


// "use client"

// import type React from "react"
// import { useEffect, useRef } from "react"

// interface YggdrasilProps {
//   className?: string
// }

// const Yggdrasil: React.FC<YggdrasilProps> = ({ className = "" }) => {
//   const containerRef = useRef<HTMLDivElement>(null)

//   useEffect(() => {
//     const container = containerRef.current
//     if (!container) return

//     // Create floating particles
//     const createParticles = () => {
//       const particlesContainer = document.createElement("div")
//       particlesContainer.className = "yggdrasil-particles"
//       for (let i = 0; i < 50; i++) {
//         const particle = document.createElement("div")
//         particle.className = "particle"
//         particle.style.setProperty("--delay", `${Math.random() * 5}s`)
//         particle.style.setProperty("--size", `${Math.random() * 3 + 1}px`)
//         particlesContainer.appendChild(particle)
//       }
//       container.appendChild(particlesContainer)
//     }

//     createParticles()

//     const handleMouseMove = (e: MouseEvent) => {
//       const { clientX, clientY } = e
//       const { width, height } = container.getBoundingClientRect()

//       const x = (clientX / width - 0.5) * 40
//       const y = (clientY / height - 0.5) * 40

//       container.style.setProperty("--mouse-x", `${x}px`)
//       container.style.setProperty("--mouse-y", `${y}px`)
//     }

//     window.addEventListener("mousemove", handleMouseMove)
//     return () => window.removeEventListener("mousemove", handleMouseMove)
//   }, [])

//   return (
//     <div ref={containerRef} className={`yggdrasil-background ${className}`.trim()}>
//       <div className="yggdrasil-cosmic-bg" />
//       <div className="yggdrasil-base" />
//       <div className="yggdrasil-trunk-glow" />
//       <div className="yggdrasil-leaves" />
//       <div className="yggdrasil-mist" />
//       <svg className="yggdrasil-filters">
//         <defs>
//           <filter id="glow">
//             <feGaussianBlur stdDeviation="8" result="coloredBlur" />
//             <feMerge>
//               <feMergeNode in="coloredBlur" />
//               <feMergeNode in="SourceGraphic" />
//             </feMerge>
//           </filter>
//         </defs>
//       </svg>
//     </div>
//   )
// }

// export default Yggdrasil


// "use client"

// import type React from "react"
// import { useEffect, useRef } from "react"

// interface YggdrasilProps {
//   className?: string
// }

// const Yggdrasil: React.FC<YggdrasilProps> = ({ className = "" }) => {
//   const containerRef = useRef<HTMLDivElement>(null)

//   useEffect(() => {
//     const container = containerRef.current
//     if (!container) return

//     const handleMouseMove = (e: MouseEvent) => {
//       const { clientX, clientY } = e
//       const { width, height } = container.getBoundingClientRect()

//       // Calculate relative position (-20 to 20)
//       const x = (clientX / width - 0.5) * 40
//       const y = (clientY / height - 0.5) * 40

//       // Update CSS variables for parallax effect
//       container.style.setProperty("--mouse-x", `${x}px`)
//       container.style.setProperty("--mouse-y", `${y}px`)
//     }

//     window.addEventListener("mousemove", handleMouseMove)
//     return () => window.removeEventListener("mousemove", handleMouseMove)
//   }, [])

//   return (
//     <div ref={containerRef} className={`yggdrasil-background ${className}`.trim()}>
//       <div className="yggdrasil-base" />
//       <div className="yggdrasil-glow" />
//       <div className="yggdrasil-particles" />
//       <div className="yggdrasil-mist" />
//     </div>
//   )
// }

// export default Yggdrasil





// "use client"
// import type React from "react"
// import { useEffect, useRef } from "react"

// const Yggdrasil: React.FC = () => {
//   const canvasRef = useRef<HTMLCanvasElement>(null)

//   useEffect(() => {
//     const canvas = canvasRef.current
//     if (!canvas) return

//     const ctx = canvas.getContext("2d")
//     if (!ctx) return

//     canvas.width = window.innerWidth
//     canvas.height = window.innerHeight

//     const drawTree = (x: number, y: number, length: number, angle: number, depth: number) => {
//       if (depth === 0) return

//       ctx.beginPath()
//       ctx.save()
//       ctx.translate(x, y)
//       ctx.rotate((angle * Math.PI) / 180)
//       ctx.moveTo(0, 0)
//       ctx.lineTo(0, -length)
//       ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 + (5 - depth) * 0.2})`
//       ctx.lineWidth = depth
//       ctx.stroke()

//       if (Math.random() < 0.5) {
//         ctx.beginPath()
//         ctx.arc(0, -length, 2, 0, Math.PI * 2)
//         ctx.fillStyle = "rgba(255, 255, 255, 0.5)"
//         ctx.fill()
//       }

//       drawTree(0, -length, length * 0.8, angle - 15, depth - 1)
//       drawTree(0, -length, length * 0.8, angle + 15, depth - 1)

//       ctx.restore()
//     }

//     const drawStar = (x: number, y: number, size: number) => {
//       ctx.beginPath()
//       ctx.arc(x, y, size, 0, Math.PI * 2)
//       ctx.fillStyle = "rgba(255, 255, 255, 0.8)"
//       ctx.fill()
//     }

//     const drawConstellation = () => {
//       const stars = []
//       for (let i = 0; i < 7; i++) {
//         stars.push({
//           x: Math.random() * canvas.width,
//           y: Math.random() * canvas.height,
//         })
//       }

//       ctx.beginPath()
//       ctx.moveTo(stars[0].x, stars[0].y)
//       for (let i = 1; i < stars.length; i++) {
//         ctx.lineTo(stars[i].x, stars[i].y)
//       }
//       ctx.strokeStyle = "rgba(255, 255, 255, 0.2)"
//       ctx.lineWidth = 1
//       ctx.stroke()

//       stars.forEach((star) => drawStar(star.x, star.y, 2))
//     }

//     const animate = () => {
//       ctx.fillStyle = "rgba(0, 0, 0, 0.05)"
//       ctx.fillRect(0, 0, canvas.width, canvas.height)

//       drawTree(canvas.width / 2, canvas.height, 120, 0, 5)

//       if (Math.random() < 0.02) {
//         drawConstellation()
//       }

//       requestAnimationFrame(animate)
//     }

//     animate()

//     const handleResize = () => {
//       canvas.width = window.innerWidth
//       canvas.height = window.innerHeight
//     }

//     window.addEventListener("resize", handleResize)

//     return () => {
//       window.removeEventListener("resize", handleResize)
//     }
//   }, [])

//   return <canvas ref={canvasRef} className="yggdrasil-canvas" />
// }

// export default Yggdrasil

