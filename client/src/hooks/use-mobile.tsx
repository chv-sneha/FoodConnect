import * as React from "react"

const MOBILE_BREAKPOINT = 768

export function useIsMobile() {
  const [isMobile, setIsMobile] = React.useState<boolean | undefined>(undefined)

  React.useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`)
    const onChange = () => {
      const mobile = window.innerWidth < MOBILE_BREAKPOINT
      const touch = 'ontouchstart' in window
      setIsMobile(mobile || touch)
    }
    mql.addEventListener("change", onChange)
    onChange() // Initial check
    window.addEventListener('orientationchange', onChange)
    return () => {
      mql.removeEventListener("change", onChange)
      window.removeEventListener('orientationchange', onChange)
    }
  }, [])

  return !!isMobile
}

export const useMobile = useIsMobile
