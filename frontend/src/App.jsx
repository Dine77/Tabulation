import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ClientList from "./components/projectList";
import CrosstabPage from "./components/CrosstabPage";  // 👈 create this file

function App() {
  return (
    <Router>
      <div className="bg-gray-50 min-h-screen">
        <Routes>
          {/* Project list page */}
          <Route path="/" element={<ClientList />} />

          {/* Crosstab page */}
          <Route path="/projects/:id/crosstabs" element={<CrosstabPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
